// cursor-claude-adapter lets Cursor BYOK use your own Anthropic-format relay.
//
// Flow: Cursor (OpenAI Chat Completions) -> this adapter converts to Anthropic
// Messages -> your upstream relay (/v1/messages) -> the Anthropic response
// (stream or not) is converted back to OpenAI -> Cursor.
//
// Cursor only lets you override the OpenAI Base URL, so the adapter takes OpenAI
// in and produces Anthropic out. It targets /v1/messages (not chat/completions)
// to keep Anthropic prompt caching and to build clean tool-call chunks itself,
// so multi-turn tool_result round-trips don't break.
//
// Config is via env vars (all have defaults); see README.md for the full setup,
// model list, thinking levels, and deployment notes.
//
// No third-party dependencies. Source is split by responsibility:
//
//	main.go     entry, HTTP handlers, config
//	convert.go  OpenAI -> Anthropic request conversion + tool-name recovery
//	stream.go   Anthropic -> OpenAI response conversion (stream + non-stream)
//	forward.go  upstream forwarding with retry/backoff
//	util.go     pure stateless helpers
package main

import (
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

// maxRequestBody caps an incoming request body (Cursor sends OpenAI Chat Completions
// JSON). It bounds memory per request so a malformed or hostile client can't force an
// unbounded read; 32 MiB leaves ample room for large multi-turn + image payloads.
const maxRequestBody = 32 << 20

var (
	upstreamURL = strings.TrimRight(env("UPSTREAM_URL", "https://api.anthropic.com"), "/") + "/v1/messages"
	modelPrefix = env("MODEL_PREFIX", "cursor-")
	anthVersion = env("ANTHROPIC_VERSION", "2023-06-01")
	port        = env("PORT", "3000")
	debug       = os.Getenv("DEBUG") == "1"
)

func main() {
	http.HandleFunc("/v1/chat/completions", handleChat)
	http.HandleFunc("/v1/responses", handleChat)
	http.HandleFunc("/v1/models", handleModels)
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) { writeJSON(w, 200, map[string]any{"status": "ok"}) })
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("cursor-claude-adapter (go, messages) ok"))
	})
	log.Printf("cursor-claude-adapter (go, /v1/messages) on :%s -> %s", port, upstreamURL)
	srv := &http.Server{Addr: ":" + port, ReadHeaderTimeout: 10 * time.Second}
	log.Fatal(srv.ListenAndServe())
}

func handleChat(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeErr(w, 405, "method not allowed")
		return
	}
	key := bearer(r)
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxRequestBody))
	if err != nil {
		if _, ok := err.(*http.MaxBytesError); ok {
			writeErr(w, 413, "request body too large")
			return
		}
		writeErr(w, 400, "read body failed")
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeErr(w, 400, "invalid json")
		return
	}

	anth, toolNames := toAnthropic(body)
	stream, _ := anth["stream"].(bool)
	model, _ := body["model"].(string)
	recoverName := makeRecover(toolNames)

	// Request summary: toolResults>0 means Cursor sent tool results back (multi-turn).
	nMsgs, nToolRes := 0, 0
	if msgs, ok := body["messages"].([]any); ok {
		nMsgs = len(msgs)
		for _, mm := range msgs {
			if m, ok := mm.(map[string]any); ok && m["role"] == "tool" {
				nToolRes++
			}
		}
	}
	log.Printf("REQ  model=%v->%v stream=%v msgs=%d toolResults=%d tools=%d toolChoice=%v",
		model, anth["model"], stream, nMsgs, nToolRes, len(toolNames), body["tool_choice"])
	out, _ := json.Marshal(anth)
	if debug {
		log.Printf("DBG  anthReq=%s", truncate(out, 2000))
	}

	resp, status, errBody := forward(r.Context(), out, key, stream)
	if resp == nil {
		log.Printf("ERR  upstream status=%d body=%s", status, truncate(errBody, 400))
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(status)
		w.Write(errBody)
		return
	}
	defer resp.Body.Close()

	if stream {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(200)
		streamToOpenAI(w, resp.Body, model, recoverName)
		return
	}
	var data map[string]any
	if b, _ := io.ReadAll(resp.Body); json.Unmarshal(b, &data) != nil {
		writeErr(w, 502, "bad upstream response")
		return
	}
	logUsage("RESP json", data["usage"])
	writeJSON(w, 200, toOpenAIResponse(data, model, recoverName))
}

// handleModels expands the model list dynamically: each baseModel yields a "no thinking"
// variant plus one variant per effort level, all with modelPrefix prepended. No config
// needed; to add a model, just edit baseModels.
// e.g. claude-opus-4-8 -> cursor-claude-opus-4-8, cursor-claude-opus-4-8-low ... -max
func handleModels(w http.ResponseWriter, r *http.Request) {
	var data []any
	add := func(id string) {
		data = append(data, map[string]any{"id": id, "object": "model", "created": 0, "owned_by": "adapter"})
	}
	for _, base := range baseModels {
		add(modelPrefix + base)
		for _, eff := range effortOrder {
			add(modelPrefix + base + "-" + eff)
		}
	}
	writeJSON(w, 200, map[string]any{"object": "list", "data": data})
}

func bearer(r *http.Request) string {
	a := r.Header.Get("Authorization")
	if len(a) > 7 && strings.EqualFold(a[:7], "Bearer ") {
		return strings.TrimSpace(a[7:])
	}
	if k := r.Header.Get("x-api-key"); k != "" {
		return k
	}
	return strings.TrimSpace(a)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

func writeErr(w http.ResponseWriter, status int, msg string) {
	writeJSON(w, status, map[string]any{"error": map[string]any{"message": msg}})
}

// errJSON wraps a raw error string as an OpenAI-style JSON error body, so the bytes we
// send on a transport failure match the application/json content-type we set.
func errJSON(msg string) []byte {
	b, _ := json.Marshal(map[string]any{"error": map[string]any{"message": msg}})
	return b
}

package main

// forward.go — upstream forwarding with retry/backoff.
// Sends the converted Anthropic request to the upstream relay and retries on
// transient failures (status codes + known relay quota/channel error bodies).

import (
	"bytes"
	"context"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

var (
	httpClient  = &http.Client{Timeout: 10 * time.Minute}
	retryStatus = map[int]bool{429: true, 500: true, 502: true, 503: true, 504: true, 524: true}
	retryDelays = []time.Duration{800 * time.Millisecond, 2 * time.Second}
)

func forward(ctx context.Context, body []byte, key string, stream bool) (*http.Response, int, []byte) {
	lastStatus, lastBody := 502, []byte(`{"error":{"message":"upstream unreachable"}}`)
	attempts := 1 + len(retryDelays)
	for attempt := 0; attempt < attempts; attempt++ {
		req, _ := http.NewRequestWithContext(ctx, http.MethodPost, upstreamURL, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", anthVersion)
		if stream {
			req.Header.Set("Accept", "text/event-stream")
		}
		resp, err := httpClient.Do(req)
		if err != nil {
			lastStatus, lastBody = 502, errJSON(err.Error())
			if ctx.Err() != nil {
				break // client gone, don't retry
			}
			sleep(attempt)
			continue
		}
		if resp.StatusCode < 400 {
			return resp, 200, nil
		}
		eb, _ := io.ReadAll(resp.Body)
		resp.Body.Close()
		lastStatus, lastBody = resp.StatusCode, eb
		if !shouldRetry(resp.StatusCode, eb) {
			break
		}
		log.Printf("upstream %d, retry %d/%d", resp.StatusCode, attempt+1, attempts)
		sleep(attempt)
	}
	return nil, lastStatus, lastBody
}

func shouldRetry(status int, body []byte) bool {
	if retryStatus[status] {
		return true
	}
	s := strings.ToLower(string(body))
	// Retry on upstream "no channel available" / quota errors. The first literal is the
	// Chinese phrase some relays return in the body — it is matched text, not a translatable comment.
	for _, kw := range []string{"无可用渠道", "no available channel", "quota", "insufficient"} {
		if strings.Contains(s, kw) {
			return true
		}
	}
	return false
}

func sleep(attempt int) {
	if attempt < len(retryDelays) {
		time.Sleep(retryDelays[attempt])
	}
}

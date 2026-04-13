package proxy

import (
	"context"
	"errors"
	"net/http"
	"testing"
)

func TestResolveRequestNoEndpointUsesRequestedPool(t *testing.T) {
	p := NewProxy(Config{DefaultPool: "default-pool"})

	_, err := p.ResolveRequest(context.Background(), ResolveRequest{
		Operation: OperationType("embed"),
		Model:     "bge-small-en-v1.5",
		Headers: map[string]string{
			"X-Termite-Pool": "requested-pool",
		},
	})
	if err == nil {
		t.Fatal("expected resolution error")
	}

	var resolutionErr *ResolutionError
	if !errors.As(err, &resolutionErr) {
		t.Fatalf("expected ResolutionError, got %T", err)
	}
	if resolutionErr.StatusCode != http.StatusServiceUnavailable {
		t.Fatalf("expected status %d, got %d", http.StatusServiceUnavailable, resolutionErr.StatusCode)
	}
	if resolutionErr.Pool != "requested-pool" {
		t.Fatalf("expected pool %q, got %q", "requested-pool", resolutionErr.Pool)
	}
}

func TestResolveRequestRateLimitUsesSingleRoutePool(t *testing.T) {
	p := NewProxy(Config{DefaultPool: "default-pool"})
	p.Router().RouteManager().AddRoute(&Route{
		Name: "rate-limited-route",
		Destinations: []Destination{
			{Pool: "routed-pool", Weight: 1},
		},
		RateLimiter: NewRateLimiter(0, 0, false),
	})

	_, err := p.ResolveRequest(context.Background(), ResolveRequest{
		Operation: OperationType("embed"),
		Model:     "bge-small-en-v1.5",
	})
	if err == nil {
		t.Fatal("expected resolution error")
	}

	var resolutionErr *ResolutionError
	if !errors.As(err, &resolutionErr) {
		t.Fatalf("expected ResolutionError, got %T", err)
	}
	if resolutionErr.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("expected status %d, got %d", http.StatusTooManyRequests, resolutionErr.StatusCode)
	}
	if resolutionErr.Pool != "routed-pool" {
		t.Fatalf("expected pool %q, got %q", "routed-pool", resolutionErr.Pool)
	}
}

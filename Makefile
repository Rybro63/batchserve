GO ?= go

.PHONY: build test race vet bench clean run

build:
	$(GO) build -o bin/server ./cmd/server
	$(GO) build -o bin/loadgen ./cmd/loadgen

test:
	$(GO) test -count=1 ./...

race:
	$(GO) test -race -count=1 ./...

vet:
	$(GO) vet ./...

run: build
	./bin/server -batch-size 32 -max-wait 5ms -workers 2 -queue 256

# Side-by-side comparison used to produce the README table.
bench: build
	@echo "=== batching disabled ==="
	@./bin/server -addr :8091 -batch-size 1 -max-wait 0 -workers 2 -queue 1024 & \
	 sleep 1.5; ./bin/loadgen -url http://localhost:8091 -rate 300 -duration 20s; kill %1
	@echo ""
	@echo "=== batching enabled ==="
	@./bin/server -addr :8092 -batch-size 32 -max-wait 5ms -workers 2 -queue 256 & \
	 sleep 1.5; ./bin/loadgen -url http://localhost:8092 -rate 300 -duration 20s; kill %1

clean:
	rm -rf bin

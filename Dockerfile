FROM golang:1.22-alpine AS build
WORKDIR /src

# Dependency manifests first, in their own layer. The project has no external
# dependencies today, so `go mod download` is a no-op — but this is the layer
# that stays cached when only source changes, and it starts working the moment
# a dependency is added. Copying go.mod and then immediately copying everything
# over it, as this did before, cached nothing.
COPY go.mod ./
RUN go mod download

COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /out/server ./cmd/server

# nonroot rather than the default: the server needs no privileges, and the
# static-debian12 base runs as root unless you ask otherwise.
FROM gcr.io/distroless/static-debian12:nonroot
COPY --from=build /out/server /server
USER nonroot:nonroot
EXPOSE 8080
ENTRYPOINT ["/server"]

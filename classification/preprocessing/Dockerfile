FROM golang:1.18-rc
WORKDIR /app
COPY . .
RUN go install
ENTRYPOINT go run preprocessing.go
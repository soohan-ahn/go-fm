FROM golang:1.9-alpine

RUN apk update && apk add git
RUN go get github.com/golang/dep/cmd/dep

#COPY Gopkg.toml /go/src/github.com/soohanboys/go-fm/
WORKDIR /go/src/github.com/soohanboys/go-fm/
#RUN dep init -vendor-only

COPY . /go/src/github.com/soohanboys/go-fm/
RUN go build -o /go/bin/go-fm
COPY /go/bin/go-fm .

CMD ["/go/bin/go-fm"]


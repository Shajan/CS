curl http://localhost:8080/hello/warp
echo ""

curl -X POST -H "Content-Type: application/json" -d '{"key": "value"}' http://localhost:8080/echo
echo ""

from fastapi import FastAPI, Request, HTTPException
# TODO: import RateLimiter from middleware

app = FastAPI()
# TODO: instantiate rate_limiter = RateLimiter()


@app.get("/data")
async def get_data(request: Request):
    client_ip = request.client.host
    # TODO: check rate_limiter.is_allowed(client_ip) and raise HTTPException(429) if not
    return {"data": "some_data", "ip": client_ip}

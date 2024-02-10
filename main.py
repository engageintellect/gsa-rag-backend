from fastapi import FastAPI

# Create a FastAPI instance
app = FastAPI()

# Define a route for the root endpoint
@app.get("/")
async def read_root():
    return {"message": "Hello, from FastAPI"}


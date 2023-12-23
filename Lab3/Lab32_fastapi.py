from fastapi import FastAPI
import uvicorn
app = FastAPI()
@app.get("/")
def root():
    return { "key":"Hello World"}
@app.post("/detect")
def root():
    return { "key":"this is the stat of driven"}
uvicorn.run(app,port=8087,host='0.0.0.0')

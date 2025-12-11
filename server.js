require("dotenv").config();
const express = require("express");
const http = require("http");
const WebSocket = require("ws");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

app.use(cors());
app.use(express.json({ limit: "50mb" }));

const PORT = process.env.PORT || 3000;
const MODEL_PATH = path.join(__dirname, "model_2");

// Load TensorFlow model
let model;
(async () => {
  try {
    model = await tf.loadLayersModel(`file://${MODEL_PATH}/model.json`);
    console.log("✅ Model loaded.");
    console.log("Input shape:", model.inputs[0].shape);
    console.log("Output shape:", model.outputs[0].shape);
  } catch (err) {
    console.error("❌ Failed to load model:", err);
  }
})();

const classNames = [
  "Oidium Heveae",
  "Healthy",
  "Anthracnose",
  "Leaf Spot",
  "Other",
];

// WebSocket connection
wss.on("connection", (ws) => {
  console.log("✅ Client connected. Total clients:", wss.clients.size);

  ws.on("message", async (message) => {
    try {
      const data = JSON.parse(message);

      if (data.type === "frame" && data.frameData) {
        if (!model) {
          ws.send(
            JSON.stringify({ type: "error", error: "Model not loaded yet" })
          );
          return;
        }

        const base64Data = data.frameData.split(",")[1];
        const buffer = Buffer.from(base64Data, "base64");

        const probabilitiesArray = await tf.tidy(() => {
          const tensor = tf.node
            .decodeImage(buffer, 3)
            .resizeNearestNeighbor([224, 224])
            .div(255.0)
            .expandDims();

          const prediction = model.predict(tensor);
          return prediction.arraySync()[0];
        });

        const predictions = probabilitiesArray.map((prob, index) => ({
          className: classNames[index],
          probability: prob,
        }));

        const topPrediction = predictions.reduce(
          (max, p) => (p.probability > max.probability ? p : max),
          { className: "", probability: 0 }
        );

        ws.send(
          JSON.stringify({
            type: "prediction",
            predictions,
            topPrediction,
            timeStamp: new Date().toISOString(),
          })
        );
      } else {
        ws.send(JSON.stringify({ type: "error", error: "Invalid message" }));
      }
    } catch (err) {
      console.error("❌ Error processing frame:", err);
      ws.send(JSON.stringify({ type: "error", error: err.message }));
    }
  });

  ws.on("close", () => {
    console.log("❌ Client disconnected. Total clients:", wss.clients.size);
  });

  ws.on("error", (err) => {
    console.error("WebSocket error:", err);
  });
});

// Optional REST endpoints for health check
app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    modelLoaded: !!model,
    memoryUsage: process.memoryUsage(),
  });
});

server.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});

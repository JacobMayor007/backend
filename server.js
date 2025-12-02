require("dotenv").config();
const express = require("express");
const multer = require("multer");
const tf = require("@tensorflow/tfjs-node");
const fs = require("fs");
const path = require("path");
const cors = require("cors");

const app = express();
app.use(cors());

const PORT = process.env.PORT || 8080;
const MODEL_PATH = path.join(__dirname, "model_2");
const UPLOAD_DIR = path.join(__dirname, "uploads");

if (!fs.existsSync(UPLOAD_DIR)) {
  fs.mkdirSync(UPLOAD_DIR, { recursive: true });
}

const upload = multer({
  dest: UPLOAD_DIR,
  limits: { fileSize: 20 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    if (file.mimetype.startsWith("image/")) {
      cb(null, true);
    } else {
      cb(new Error("Only image files are allowed!"), false);
    }
  },
});

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

// Predict endpoint
app.post("/predict", upload.single("image"), async (req, res) => {
  if (!model) {
    return res
      .status(503)
      .json({ error: "Model not loaded yet. Please try again later." });
  }

  if (!req.file) {
    return res.status(400).json({ error: "No image uploaded" });
  }

  const imagePath = req.file.path;

  try {
    console.log(
      `📸 Processing image: ${req.file.originalname || req.file.filename}`
    );

    const buffer = fs.readFileSync(imagePath);

    // Wrap ALL tensor operations in tf.tidy() for automatic cleanup
    const probabilitiesArray = await tf.tidy(() => {
      const tensor = tf.node
        .decodeImage(buffer, 3)
        .resizeNearestNeighbor([224, 224])
        .div(255.0)
        .expandDims();

      console.log("📐 Tensor shape:", tensor.shape);

      let prediction;
      try {
        prediction = model.predict(tensor);
      } catch (err) {
        throw new Error("Model prediction failed: " + err.message);
      }

      // Return the array data to use outside tidy
      return prediction.arraySync()[0];
    });

    console.log("📊 Raw probabilities:", probabilitiesArray);

    const classNames = [
      "Oidium Heveae",
      "Healthy",
      "Anthracnose",
      "Leaf Spot",
      "Other",
    ];

    if (probabilitiesArray.length !== classNames.length) {
      throw new Error(
        `Expected ${classNames.length} output classes but got ${probabilitiesArray.length}`
      );
    }

    const predictions = probabilitiesArray.map((prob, index) => ({
      className: classNames[index],
      probability: prob,
    }));

    const topPrediction = predictions.reduce(
      (max, p) => (p.probability > max.probability ? p : max),
      { className: "", probability: 0 }
    );

    if (topPrediction.probability < 0.6) {
      console.warn("⚠️ Low confidence prediction:", topPrediction);
    }

    const response = {
      tfjsVersion: "1.7.4",
      tmVersion: "2.4.10",
      modelName: "tm-my-image-model",
      labels: classNames,
      predictions,
      topPrediction,
      imageSize: 224,
      timeStamp: new Date().toISOString(),
    };

    console.log("✅ Top prediction:", topPrediction);

    // Delete uploaded image
    try {
      await fs.promises.unlink(imagePath);
    } catch (err) {
      console.error("❌ Error deleting uploaded file:", err);
    }

    return res.json(response);
  } catch (err) {
    console.error("❌ Prediction error:", err.message);

    // Cleanup on failure
    if (fs.existsSync(imagePath)) {
      try {
        await fs.promises.unlink(imagePath);
      } catch (err2) {
        console.error("❌ Failed to delete file after error:", err2.message);
      }
    }

    return res.status(500).json({
      error: "Prediction failed",
      details: err.message,
    });
  }
});

app.get("/predict", (req, res) => {
  res.send(`✅ Server is running and listening on port ${PORT}`);
});

app.get("/health", (req, res) => {
  res.json({
    status: "ok",
    modelLoaded: !!model,
    memoryUsage: process.memoryUsage(),
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server running at http://localhost:${PORT}`);
});

const statusEl = document.getElementById("status");
const checkpointEl = document.getElementById("checkpoint");
const promptEl = document.getElementById("prompt");
const guidanceEl = document.getElementById("guidance-scale");
const samplePromptsEl = document.getElementById("sample-prompts");
const runInfoEl = document.getElementById("run-info-list");
const generationMetaEl = document.getElementById("generation-meta");
const formEl = document.getElementById("generator-form");
const buttonEl = document.getElementById("generate-button");
const resultImageEl = document.getElementById("result-image");
const emptyStateEl = document.getElementById("empty-state");

let checkpointMap = new Map();

function setStatus(message) {
  statusEl.textContent = message;
}

function formatMetaValue(value) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }
  return String(value);
}

function renderDefinitionList(target, data) {
  target.innerHTML = "";
  Object.entries(data).forEach(([key, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = key;
    const dd = document.createElement("dd");
    dd.textContent = formatMetaValue(value);
    target.append(dt, dd);
  });
}

function renderSamplePrompts(prompts) {
  samplePromptsEl.innerHTML = "";
  prompts.forEach((prompt) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "chip";
    button.textContent = prompt;
    button.addEventListener("click", () => {
      promptEl.value = prompt;
    });
    samplePromptsEl.appendChild(button);
  });
}

function updateCheckpointContext() {
  const selected = checkpointMap.get(checkpointEl.value);
  if (!selected) {
    renderDefinitionList(runInfoEl, {});
    renderSamplePrompts([]);
    return;
  }

  guidanceEl.placeholder = `Default ${selected.guidance_scale}`;
  renderDefinitionList(runInfoEl, {
    "Run id": selected.run_id,
    "Image size": `${selected.image_size}x${selected.image_size}`,
    "Timesteps": selected.timesteps,
    "Default guidance": selected.guidance_scale,
  });
  renderSamplePrompts(selected.training_prompts || []);
}

async function loadCheckpoints() {
  setStatus("Loading checkpoints...");
  const response = await fetch("/api/checkpoints");
  const payload = await response.json();

  checkpointMap = new Map(
    payload.checkpoints.map((checkpoint) => [checkpoint.checkpoint_path, checkpoint]),
  );

  checkpointEl.innerHTML = "";
  payload.checkpoints.forEach((checkpoint) => {
    const option = document.createElement("option");
    option.value = checkpoint.checkpoint_path;
    option.textContent = `${checkpoint.run_id} · ${checkpoint.timesteps} steps`;
    checkpointEl.appendChild(option);
  });

  if (payload.default_checkpoint) {
    checkpointEl.value = payload.default_checkpoint;
    updateCheckpointContext();
    setStatus("Ready to generate.");
  } else {
    setStatus("No checkpoints found yet. Train a model first.");
  }
}

checkpointEl.addEventListener("change", updateCheckpointContext);

formEl.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!checkpointEl.value) {
    setStatus("No checkpoint available yet.");
    return;
  }

  buttonEl.disabled = true;
  setStatus("Generating image from noise. This may take a few seconds.");

  const body = {
    prompt: promptEl.value.trim(),
    checkpoint_path: checkpointEl.value,
  };

  if (guidanceEl.value.trim()) {
    body.guidance_scale = Number(guidanceEl.value);
  }

  try {
    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Generation failed.");
    }

    const payload = await response.json();
    resultImageEl.src = `data:image/png;base64,${payload.image_base64}`;
    resultImageEl.hidden = false;
    emptyStateEl.hidden = true;

    renderDefinitionList(generationMetaEl, {
      Prompt: payload.prompt,
      "Saved file": payload.saved_path,
      "Run id": payload.meta.run_id,
      Guidance: payload.meta.guidance_scale,
      Timesteps: payload.meta.timesteps,
      Size: `${payload.meta.image_size}x${payload.meta.image_size}`,
    });
    setStatus("Image generated successfully.");
  } catch (error) {
    setStatus(error.message);
  } finally {
    buttonEl.disabled = false;
  }
});

loadCheckpoints().catch((error) => {
  setStatus(`Failed to load checkpoints: ${error.message}`);
});

const state = {
  sessionId: null,
  referenceId: null,
  turns: [],
  roundCounter: 0,
  suggestedTexts: [],
  suggestionRequestId: 0,
};

const galleryEl = document.getElementById("gallery");
const roundsEl = document.getElementById("rounds");
const turnsEl = document.getElementById("turns");
const sessionInfoEl = document.getElementById("sessionInfo");
const statusEl = document.getElementById("status");
const referenceCodeEl = document.getElementById("referenceCode");
const applyReferenceBtnEl = document.getElementById("applyReferenceBtn");

const resetBtnEl = document.getElementById("resetBtn");

function setStatus(text) {
  statusEl.textContent = text;
}

function setRoundStatus(roundEl, text) {
  const inlineStatus = roundEl?.querySelector(".inline-status");
  if (inlineStatus) {
    inlineStatus.textContent = text;
    return;
  }
  setStatus(text);
}

function renderSession() {
  sessionInfoEl.textContent = state.referenceId
    ? `reference=${state.referenceId}`
    : "尚未选择参考图";

  turnsEl.innerHTML = "";
  state.turns.forEach((turn, idx) => {
    const li = document.createElement("li");
    li.textContent = `Turn ${idx + 1}: ${turn}`;
    turnsEl.appendChild(li);
  });
}

function setReference(imageId) {
  state.referenceId = imageId;
  renderSession();
  refreshSelectableBorders();
  updateSuggestionsForReference(imageId);
}

function renderSuggestionOptions(roundBlock, suggestions) {
  if (!roundBlock) return;
  const host = roundBlock.querySelector(".suggestion-options");
  const textarea = roundBlock.querySelector("textarea");
  if (!host || !textarea) return;

  host.innerHTML = "";
  const values = Array.isArray(suggestions) ? suggestions.slice(0, 2) : [];

  if (values.length === 0) {
    const empty = document.createElement("span");
    empty.className = "suggestion-empty";
    empty.textContent = "暂无推荐文本";
    host.appendChild(empty);
    return;
  }

  values.forEach((text) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "suggestion-chip";
    btn.textContent = text;
    btn.addEventListener("click", () => {
      textarea.value = text;
      textarea.focus();
    });
    host.appendChild(btn);
  });
}

function renderActiveRoundSuggestions() {
  const activeRound = roundsEl.querySelector(".round-block.active");
  if (!activeRound) return;
  renderSuggestionOptions(activeRound, state.suggestedTexts);
}

async function updateSuggestionsForReference(imageId) {
  if (!imageId) {
    state.suggestedTexts = [];
    renderActiveRoundSuggestions();
    return;
  }

  const requestId = ++state.suggestionRequestId;
  try {
    const resp = await fetch(`/api/reference/${encodeURIComponent(imageId)}`);
    if (!resp.ok) {
      throw new Error(await resp.text());
    }

    const data = await resp.json();
    if (requestId !== state.suggestionRequestId) {
      return;
    }

    state.suggestedTexts = Array.isArray(data.suggested_texts)
      ? data.suggested_texts.slice(0, 2)
      : [];
    renderActiveRoundSuggestions();
  } catch (_err) {
    if (requestId !== state.suggestionRequestId) {
      return;
    }
    state.suggestedTexts = [];
    renderActiveRoundSuggestions();
  }
}

function createImageCard(imageId, imageUrl, onClick, extraText = "") {
  const card = document.createElement("div");
  card.className = "card";
  if (state.referenceId === imageId) {
    card.classList.add("selected");
  }

  const img = document.createElement("img");
  img.src = imageUrl;
  img.alt = imageId;

  const caption = document.createElement("div");
  caption.className = "caption";
  caption.textContent = imageId;

  card.appendChild(img);
  card.appendChild(caption);

  if (extraText) {
    const meta = document.createElement("div");
    meta.className = "result-meta";
    meta.textContent = extraText;
    card.appendChild(meta);
  }

  card.addEventListener("click", () => onClick(imageId));
  return card;
}

function pinRoundResult(roundBlockEl, item) {
  const roundResultsEl = roundBlockEl.querySelector(".round-results");
  if (!roundResultsEl) return;

  roundResultsEl.innerHTML = "";
  const pinnedCard = createImageCard(
    item.image_id,
    item.image_url,
    () => {},
    `score=${item.score.toFixed(4)} | 已选为下一轮参考图`,
  );
  pinnedCard.classList.add("selected");
  roundResultsEl.appendChild(pinnedCard);
}

function collapseGalleryToSingle(imageId) {
  if (!imageId) return;
  const pinned = createImageCard(imageId, `/images/${imageId}.jpg`, () => {
    setReference(imageId);
  });
  pinned.classList.add("selected");
  galleryEl.innerHTML = "";
  galleryEl.appendChild(pinned);
}

async function applyReferenceByCode() {
  const raw = (referenceCodeEl?.value || "").trim();
  if (!raw) {
    setStatus("请先输入参考图编码");
    return;
  }

  const imageId = raw.toUpperCase();
  try {
    const resp = await fetch(`/api/reference/${encodeURIComponent(imageId)}`);
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    if (!data.exists) {
      setStatus(`未找到参考图编码: ${imageId}，请检查后重试`);
      return;
    }

    setReference(data.image_id);
    collapseGalleryToSingle(data.image_id);
    setStatus(`已设置首轮参考图: ${data.image_id}`);
  } catch (err) {
    setStatus(`设置参考图失败: ${String(err)}`);
  }
}

function pinLastCompletedRoundIfSelected() {
  const completed = Array.from(roundsEl.querySelectorAll(".round-block:not(.active)"));
  if (completed.length === 0) return;
  const lastCompleted = completed[completed.length - 1];
  if (lastCompleted.dataset.pinned === "1") return;
  if (!lastCompleted._selectedItem) return;

  pinRoundResult(lastCompleted, lastCompleted._selectedItem);
  lastCompleted.dataset.pinned = "1";
}

function previousRoundNeedsReferenceSelection() {
  const completed = Array.from(roundsEl.querySelectorAll(".round-block:not(.active)"));
  if (completed.length === 0) {
    return false;
  }
  const lastCompleted = completed[completed.length - 1];
  return !lastCompleted._selectedItem;
}

async function loadGallery() {
  const resp = await fetch("/api/gallery");
  const data = await resp.json();

  galleryEl.innerHTML = "";
  data.items.forEach((item) => {
    const card = createImageCard(item.image_id, item.image_url, (imageId) => {
      setReference(imageId);
    });
    galleryEl.appendChild(card);
  });

  if (!state.referenceId && data.items.length > 0) {
    setReference(data.items[0].image_id);
  } else {
    renderSession();
    refreshSelectableBorders();
  }
}

function refreshSelectableBorders() {
  document.querySelectorAll(".card").forEach((node) => {
    const caption = node.querySelector(".caption");
    if (!caption) return;
    const id = caption.textContent;
    if (id === state.referenceId) {
      node.classList.add("selected");
    } else {
      node.classList.remove("selected");
    }
  });
}

async function ensureSession() {
  if (state.sessionId) return;
  const resp = await fetch("/api/session/new", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ reference_id: state.referenceId }),
  });
  if (!resp.ok) {
    throw new Error(await resp.text());
  }
  const data = await resp.json();
  state.sessionId = data.session_id;
  setReference(data.reference_id);
  state.turns = data.turns;
  renderSession();
}

async function runRetrieve() {
  const activeRound = roundsEl.querySelector(".round-block.active");
  if (!activeRound) {
    setStatus("未找到可输入的新轮次");
    return;
  }

  const modifiedTextEl = activeRound.querySelector("textarea");
  const topKEl = activeRound.querySelector("input[type='number']");
  const resultsEl = activeRound.querySelector(".round-results");

  const text = modifiedTextEl.value.trim();
  if (!text) {
    setRoundStatus(activeRound, "请先输入 modified text");
    return;
  }

  if (!state.referenceId) {
    setRoundStatus(activeRound, "请先选择参考图像");
    return;
  }

  if (previousRoundNeedsReferenceSelection()) {
    setRoundStatus(activeRound, "请先在上一轮结果中选择下一轮参考图像");
    return;
  }

  try {
    setRoundStatus(activeRound, "检索中...");
    pinLastCompletedRoundIfSelected();
    await ensureSession();

    const referenceUsedThisRound = state.referenceId;

    const payload = {
      modified_text: text,
      top_k: Number(topKEl.value || 10),
      reference_id: state.referenceId,
    };

    const resp = await fetch(`/api/session/${state.sessionId}/retrieve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!resp.ok) {
      throw new Error(await resp.text());
    }

    const data = await resp.json();
    setReference(data.reference_id);
    state.turns = data.turns;

    // After first retrieval, keep only the selected reference image in gallery.
    if (state.roundCounter === 1) {
      collapseGalleryToSingle(referenceUsedThisRound);
    }

    resultsEl.innerHTML = "";
    data.results.forEach((item) => {
      const card = createImageCard(
        item.image_id,
        item.image_url,
        (imageId) => {
          setReference(item.image_id);
          activeRound._selectedItem = item;
          resultsEl.querySelectorAll(".card").forEach((node) => node.classList.remove("selected"));
          card.classList.add("selected");
          setRoundStatus(activeRound, `已选择下一轮参考图: ${imageId}`);
        },
        `score=${item.score.toFixed(4)}`,
      );
      resultsEl.appendChild(card);
    });

    activeRound.classList.remove("active");
    modifiedTextEl.readOnly = true;
    const retrieveBtnEl = activeRound.querySelector("button");
    if (retrieveBtnEl) {
      retrieveBtnEl.disabled = true;
    }

    appendNewRoundComposer();
    setRoundStatus(activeRound, `完成: 返回 ${data.results.length} 条结果`);
    renderSession();
    refreshSelectableBorders();
  } catch (err) {
    setRoundStatus(activeRound, `错误: ${String(err)}`);
  }
}

async function resetSession() {
  if (!state.sessionId) {
    state.turns = [];
    await loadGallery();
    roundsEl.innerHTML = "";
    state.roundCounter = 0;
    appendNewRoundComposer();
    renderSession();
    setStatus("会话未创建，已清空本地历史");
    return;
  }

  try {
    const resp = await fetch(`/api/session/${state.sessionId}/reset`, { method: "POST" });
    if (!resp.ok) {
      throw new Error(await resp.text());
    }
    const data = await resp.json();
    state.turns = data.turns;
    state.sessionId = null;
    await loadGallery();
    roundsEl.innerHTML = "";
    state.roundCounter = 0;
    appendNewRoundComposer();
    setStatus("会话已重置");
    renderSession();
  } catch (err) {
    setStatus(`重置失败: ${String(err)}`);
  }
}

resetBtnEl.addEventListener("click", resetSession);
applyReferenceBtnEl.addEventListener("click", applyReferenceByCode);
referenceCodeEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    applyReferenceByCode();
  }
});

function appendNewRoundComposer() {
  state.roundCounter += 1;

  const block = document.createElement("div");
  block.className = "round-block active";

  const head = document.createElement("div");
  head.className = "round-head";
  head.textContent = `Round ${state.roundCounter} 输入文本并检索`;

  const rowText = document.createElement("div");
  rowText.className = "control-row";

  const rowSuggestion = document.createElement("div");
  rowSuggestion.className = "control-row";
  const suggestionLabel = document.createElement("div");
  suggestionLabel.className = "suggestion-label";
  suggestionLabel.textContent = "推荐描述（点击可自动填入）";
  const suggestionOptions = document.createElement("div");
  suggestionOptions.className = "suggestion-options";
  rowSuggestion.appendChild(suggestionLabel);
  rowSuggestion.appendChild(suggestionOptions);

  const textarea = document.createElement("textarea");
  textarea.placeholder = "例如: make it sleeveless and brighter";
  rowText.appendChild(textarea);

  const rowInline = document.createElement("div");
  rowInline.className = "control-row inline";

  const topkLabel = document.createElement("label");
  topkLabel.textContent = "Top K";
  const topkInput = document.createElement("input");
  topkInput.type = "number";
  topkInput.min = "1";
  topkInput.max = "200";
  topkInput.value = "10";

  const retrieveBtn = document.createElement("button");
  retrieveBtn.textContent = "Retrieve";
  retrieveBtn.addEventListener("click", runRetrieve);

  const inlineStatus = document.createElement("span");
  inlineStatus.className = "inline-status";

  rowInline.appendChild(topkLabel);
  rowInline.appendChild(topkInput);
  rowInline.appendChild(retrieveBtn);
  rowInline.appendChild(inlineStatus);

  const resultTitle = document.createElement("div");
  resultTitle.className = "round-head";
  resultTitle.textContent = `Round ${state.roundCounter} 检索结果`;

  const resultGrid = document.createElement("div");
  resultGrid.className = "round-results";

  block.appendChild(head);
  block.appendChild(rowSuggestion);
  block.appendChild(rowText);
  block.appendChild(rowInline);
  block.appendChild(resultTitle);
  block.appendChild(resultGrid);

  roundsEl.appendChild(block);
  renderSuggestionOptions(block, state.suggestedTexts);
  textarea.focus();
}

loadGallery()
  .then(() => appendNewRoundComposer())
  .catch((err) => setStatus(`加载图库失败: ${String(err)}`));

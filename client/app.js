let currentAtBatData = [];
let currentStand = "R";
let allAtBats = {};
let allAtBatsMeta = {};
let allPitchersData = {};
let currentArsenalHTML = "";
let tsPitcher = null;
let tsBatter = null;
let currentAdvStats = {};

// 🌟 雲端專屬：自動判斷目前是「本地開發」還是「正式上線」
const API_BASE_URL =
  window.location.hostname === "127.0.0.1" ||
  window.location.hostname === "localhost"
    ? "http://127.0.0.1:5000"
    : "https://baseball-web-game.onrender.com";

// 🌟 本地端最新移植：完美 17-50-83 座標網格
const zoneCoordinates = {
  左上: { x: 17, y: 17 },
  中上: { x: 50, y: 17 },
  右上: { x: 83, y: 17 },
  左中: { x: 17, y: 50 },
  正中: { x: 50, y: 50 },
  右中: { x: 83, y: 50 },
  左下: { x: 17, y: 83 },
  中下: { x: 50, y: 83 },
  右下: { x: 83, y: 83 },
  壞_左上: { x: -17, y: -17 },
  壞_中上: { x: 50, y: -17 },
  壞_右上: { x: 117, y: -17 },
  壞_左中: { x: -17, y: 50 },
  壞_右中: { x: 117, y: 50 },
  壞_左下: { x: -17, y: 117 },
  壞_中下: { x: 50, y: 117 },
  壞_右下: { x: 117, y: 117 },
  壞_挖地瓜: { x: 50, y: 134 },
};

async function loadAllPitchers() {
  try {
    const response = await fetch(API_BASE_URL + "/api/pitchers");
    allPitchersData = await response.json();

    const pitcherSelect = document.getElementById("pitcherSelect");
    pitcherSelect.innerHTML = "";

    const pIds = Object.keys(allPitchersData).sort((a, b) =>
      a === "808967" ? -1 : 1
    );

    pIds.forEach((pId) => {
      const opt = document.createElement("option");
      opt.value = pId;
      opt.innerText = allPitchersData[pId].name;
      pitcherSelect.appendChild(opt);
    });

    tsPitcher = new TomSelect("#pitcherSelect", {
      create: false,
      sortField: false,
      placeholder: "請輸入英文搜尋投手...",
    });

    tsPitcher.on("change", (value) => {
      if (value) updatePitcherUI(value);
    });

    tsPitcher.setValue("808967", true);
    updatePitcherUI("808967");
  } catch (error) {
    console.error("無法載入投手清單", error);
  }
}

function updatePitcherUI(pitcherId) {
  const typeSelect = document.getElementById("newPitchType");
  typeSelect.innerHTML = "";
  currentArsenalHTML = "";

  if (
    allPitchersData[pitcherId] &&
    allPitchersData[pitcherId].arsenal.length > 0
  ) {
    allPitchersData[pitcherId].arsenal.forEach((pitch) => {
      const optStr = `<option value="${pitch.value}">${pitch.label}</option>`;
      currentArsenalHTML += optStr;
      const opt = document.createElement("option");
      opt.value = pitch.value;
      opt.innerText = pitch.label;
      typeSelect.appendChild(opt);
    });
  }
  loadAtBats(pitcherId);
  buildSequenceUI();
}

async function loadAtBats(pitcherId) {
  try {
    const response = await fetch(API_BASE_URL + "/api/atbats/" + pitcherId);
    const atbats = await response.json();

    if (tsBatter) tsBatter.destroy();

    const batterSelect = document.getElementById("batterSelect");
    batterSelect.innerHTML = "";

    if (atbats.length === 0) {
      batterSelect.innerHTML = '<option value="">(無符合條件的打席)</option>';
      return;
    }

    atbats.forEach((ab) => {
      allAtBats[ab.ab_id] = ab.sequence;
      allAtBatsMeta[ab.ab_id] = ab.stand;
      const opt = document.createElement("option");
      opt.value = ab.ab_id;
      opt.innerText = ab.label;
      batterSelect.appendChild(opt);
    });

    tsBatter = new TomSelect("#batterSelect", {
      create: false,
      sortField: false,
      placeholder: "請輸入英文搜尋打者與日期...",
      maxOptions: 1000,
    });

    tsBatter.on("change", (value) => {
      if (value) {
        currentAtBatData = allAtBats[value];
        currentStand = allAtBatsMeta[value];

        const selectedAb = atbats.find((ab) => ab.ab_id === value);
        currentAdvStats =
          selectedAb && selectedAb.adv_stats ? selectedAb.adv_stats : {};
        console.log(`\n✅ 成功載入打席: ${selectedAb.label}`);
        console.table(currentAdvStats);

        renderOriginal();
      }
    });

    if (atbats.length > 0) {
      const firstId = atbats[0].ab_id;
      tsBatter.setValue(firstId, true);
      currentAtBatData = allAtBats[firstId];
      currentStand = allAtBatsMeta[firstId];

      const firstAb = atbats[0];
      currentAdvStats = firstAb.adv_stats || {};
      console.log(`\n✅ 預設載入打席: ${firstAb.label}`);
      console.table(currentAdvStats);

      renderOriginal();
    }
  } catch (error) {
    console.error("無法載入打席資料:", error);
  }
}

function renderOriginal() {
  const zone = document.getElementById("originalZone");
  const list = document.getElementById("originalList");
  zone.innerHTML = '<div class="zone-cell"></div>'.repeat(9);
  list.innerHTML = "";

  currentAtBatData.forEach((pitch) => {
    const dot = document.createElement("div");
    dot.className = `pitch-dot ${pitch.isStrike ? "strike-dot" : "ball-dot"}`;
    dot.style.left = `${pitch.x}%`;
    dot.style.top = `${pitch.y}%`;
    dot.innerText = pitch.num;
    zone.appendChild(dot);

    let resultText = pitch.result;
    if (pitch.result === "hit_into_play" && pitch.launch_speed) {
      resultText = `${pitch.result} <span style="color: #e74c3c; font-size: 0.9em; font-weight: bold; margin-left: 6px;">🚀 EV: ${pitch.launch_speed}</span>`;
    }

    const item = document.createElement("div");
    item.className = `pitch-item ${pitch.isStrike ? "strike" : "ball"}`;
    item.innerHTML = `
            <div class="pitch-number">${pitch.num}</div>
            <div class="pitch-info">
                <div class="pitch-result">${resultText}</div>
                <div class="pitch-details"><strong>${pitch.speed}</strong> ${pitch.name}</div>
            </div>
            <div class="pitch-count">${pitch.count}</div>
        `;
    list.appendChild(item);
  });

  if (currentAtBatData.length > 0) {
    renderCounterfactual({
      x: 50,
      y: 140,
      type: document.getElementById("newPitchType").value,
    });
  }
}

function renderCounterfactual(newPitchData) {
  const zone = document.getElementById("cfZone");
  const list = document.getElementById("cfList");
  const simMode = document.getElementById("simMode").value;

  zone.innerHTML = '<div class="zone-cell"></div>'.repeat(9);

  currentAtBatData.forEach((p, idx) => {
    const isLast = idx === currentAtBatData.length - 1;
    const isSecondLast = idx === currentAtBatData.length - 2;

    let x = p.x;
    let y = p.y;
    let isModified = false;

    if (simMode === "last" && isLast) {
      x = newPitchData.x;
      y = newPitchData.y;
      isModified = true;
    } else if (simMode === "second_last" && isSecondLast) {
      x = newPitchData.x;
      y = newPitchData.y;
      isModified = true;
    }

    const dot = document.createElement("div");
    dot.className = `pitch-dot ${
      isModified ? "cf-dot" : p.isStrike ? "strike-dot" : "ball-dot"
    }`;
    dot.style.left = `${x}%`;
    dot.style.top = `${y}%`;
    dot.innerText = p.num;

    if (simMode === "last" && !isLast) dot.style.opacity = "0.3";
    if (simMode === "second_last" && idx < currentAtBatData.length - 2)
      dot.style.opacity = "0.3";

    zone.appendChild(dot);
  });

  const targetZoneName =
    document.getElementById("newPitchZone").options[
      document.getElementById("newPitchZone").selectedIndex
    ].text;
  const modifiedNum =
    simMode === "last" ? currentAtBatData.length : currentAtBatData.length - 1;

  list.innerHTML = `
      <div class="pitch-item modified">
          <div class="pitch-number" style="background:var(--highlight)">${modifiedNum}</div>
          <div class="pitch-info">
              <div class="pitch-result" style="color:var(--highlight)">⚠️ 變更戰術：第 ${modifiedNum} 球改投</div>
              <div class="pitch-details"><strong>${
                newPitchData.type || "N/A"
              }</strong> (${targetZoneName})</div>
          </div>
      </div>
      <div style="font-size: 13px; color: #666; margin-top: 10px; padding-left: 10px;">
          ℹ️ ${
            simMode === "last"
              ? "觀察決戰球本身的壓制效益"
              : "觀察「佈局球改變」引發的 Delta 視線干擾，對最後一球產生的期望值影響。"
          }
      </div>
  `;
}

async function runSimulation() {
  const simMode = document.getElementById("simMode").value;
  if (simMode === "second_last" && currentAtBatData.length < 2) {
    alert("此打席只有一球，無法改變佈局球 (N-1)！");
    document.getElementById("simMode").value = "last";
    return;
  }

  const pType = document.getElementById("newPitchType").value;
  const pZone = document.getElementById("newPitchZone").value;
  const coords = zoneCoordinates[pZone] || { x: 50, y: 50 };

  const btn = document.querySelectorAll(".btn-simulate")[0];
  btn.innerText = "神經網路運算中...";
  btn.disabled = true;

  const getPhys = (pitchObj) => ({
    speed: parseFloat(pitchObj.raw_speed) / 100.0,
    px: pitchObj.raw_px,
    pz: pitchObj.raw_pz,
    pfx_x: pitchObj.raw_pfx_x,
    pfx_z: pitchObj.raw_pfx_z,
  });

  let n2_data = { speed: 0.95, px: 0, pz: 0, pfx_x: 0, pfx_z: 0 };
  let n1_data = { speed: 0.95, px: 0, pz: 0, pfx_x: 0, pfx_z: 0 };
  let orig_data = { speed: 0.95, px: 0, pz: 0, pfx_x: 0, pfx_z: 0 };
  let n1_out = [1, 0, 0];

  const len = currentAtBatData.length;
  if (len >= 3) n2_data = getPhys(currentAtBatData[len - 3]);
  if (len >= 2) {
    const n1 = currentAtBatData[len - 2];
    n1_data = getPhys(n1);
    const res = n1.result;
    const isTake = [
      "called_strike",
      "ball",
      "blocked_ball",
      "pitchout",
    ].includes(res)
      ? 1
      : 0;
    const isWhiff = [
      "swinging_strike",
      "swinging_pitchout",
      "swinging_strike_blocked",
      "foul_tip",
    ].includes(res)
      ? 1
      : 0;
    const isFoul = ["foul", "foul_bunt"].includes(res) ? 1 : 0;
    n1_out = [isTake, isWhiff, isFoul];
  }

  let bCount = 0;
  let sCount = 0;
  if (len >= 1) {
    const countStr = currentAtBatData[len - 1].count;
    const parts = countStr.split("-");
    if (parts.length === 2) {
      bCount = parseInt(parts[0].trim());
      sCount = parseInt(parts[1].trim());
    }
  }
  if (len >= 1) orig_data = getPhys(currentAtBatData[len - 1]);

  try {
    const response = await fetch(API_BASE_URL + "/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pitcher_id: document.getElementById("pitcherSelect").value,
        sim_mode: simMode,
        pitch_type: pType,
        pitch_zone: pZone,
        stand: currentStand,
        adv_stats: currentAdvStats,
        b_count: bCount,
        s_count: sCount,
        n2_pitch: n2_data,
        n1_pitch: n1_data,
        n1_outcome: n1_out,
        orig_pitch: orig_data,
      }),
    });

    // 🌟 雲端專屬：攔截錯誤狀態碼與 429 防護
    if (!response.ok) {
      if (response.status === 429) {
        alert("⏳ 您的運算請求太頻繁了，請稍等一分鐘後再試！");
      } else if (response.status === 500) {
        alert("💥 伺服器運算發生錯誤，請重新整理頁面。");
      } else {
        alert(`⚠️ 發生未知錯誤 (HTTP ${response.status})`);
      }
      return;
    }

    const result = await response.json();
    renderCounterfactual({ x: coords.x, y: coords.y, type: pType });

    const labels = [
      { id: "Take", idx: 0, isPitcherFriendly: null },
      { id: "Whiff", idx: 1, isPitcherFriendly: true },
      { id: "Foul", idx: 2, isPitcherFriendly: true },
      { id: "Weak", idx: 3, isPitcherFriendly: true },
      { id: "Hard", idx: 4, isPitcherFriendly: false },
    ];

    labels.forEach((item) => {
      const origEl = document.getElementById(`orig${item.id}`);
      const newEl = document.getElementById(`new${item.id}`);
      if (!origEl || !newEl) return;

      const origVal = result.orig_probs[item.idx];
      const newVal = result.cf_probs[item.idx];

      origEl.style.display = "inline";
      origEl.innerText = `${origVal}%`;
      newEl.innerText = `${newVal}%`;

      if (item.isPitcherFriendly !== null && newVal !== origVal) {
        const isImproved = item.isPitcherFriendly
          ? newVal > origVal
          : newVal < origVal;
        newEl.className = `val-new ${
          isImproved ? "val-improve" : "val-worsen"
        }`;
      } else {
        newEl.className = "val-new";
      }
    });
  } catch (error) {
    console.error(error);
  } finally {
    btn.innerText = "重新模擬打席";
    btn.disabled = false;
  }
}

window.onload = () => {
  loadAllPitchers();
};

function buildSequenceUI() {
  const container = document.getElementById("sequenceBuilderContainer");
  const length = parseInt(document.getElementById("seqLength").value) || 5;
  container.innerHTML = "";

  const sourceZones = document.getElementById("newPitchZone").innerHTML;

  for (let i = 1; i <= length; i++) {
    const row = document.createElement("div");
    row.style.display = "flex";
    row.style.gap = "10px";
    row.style.alignItems = "center";

    row.innerHTML = `
          <div style="width: 25px; height: 25px; background: #e67e22; color: white; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; font-size: 12px;">${i}</div>
          <select id="seqType_${i}" style="flex: 1;">${
      currentArsenalHTML || '<option value="">載入中...</option>'
    }</select>
          <select id="seqZone_${i}" style="flex: 1;">${sourceZones}</select>
      `;
    container.appendChild(row);
  }
}

async function runSequenceSimulation() {
  const length = parseInt(document.getElementById("seqLength").value) || 5;
  const pitcherId = document.getElementById("pitcherSelect").value;

  const pitches = [];
  for (let i = 1; i <= length; i++) {
    pitches.push({
      type: document.getElementById(`seqType_${i}`).value,
      zone: document.getElementById(`seqZone_${i}`).value,
    });
  }

  const btn = document.getElementById("btnSequence");
  btn.innerText = "運算中...";
  btn.disabled = true;

  try {
    const response = await fetch(API_BASE_URL + "/api/simulate_sequence", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pitcher_id: pitcherId,
        stand: currentStand,
        adv_stats: currentAdvStats,
        pitches: pitches,
      }),
    });

    // 🌟 雲端專屬：攔截錯誤狀態碼與 429 防護
    if (!response.ok) {
      if (response.status === 429) {
        alert("⏳ 您使用的算力已達上限，請稍等一分鐘後再試！");
      } else if (response.status === 500) {
        alert("💥 模型運算發生錯誤，可能是極端球種參數異常。");
      } else {
        alert(`⚠️ 發生未知錯誤 (HTTP ${response.status})`);
      }
      return;
    }

    const data = await response.json();

    const zone = document.getElementById("sequenceZone");
    zone.innerHTML = '<div class="zone-cell"></div>'.repeat(9);

    data.results.forEach((r, idx) => {
      const coords = zoneCoordinates[r.zone] || { x: 50, y: 50 };
      const dot = document.createElement("div");
      const isLast = idx === data.results.length - 1;

      dot.className = `pitch-dot ${isLast ? "cf-dot" : "strike-dot"}`;
      dot.style.left = `${coords.x}%`;
      dot.style.top = `${coords.y}%`;
      if (!isLast) dot.style.opacity = "0.5";
      if (isLast) dot.style.background = "#e67e22";
      dot.innerText = r.pitch_num;

      zone.appendChild(dot);
    });

    const list = document.getElementById("sequenceResultsList");
    list.innerHTML = "";

    data.results.forEach((r) => {
      const item = document.createElement("div");
      item.className = "pitch-item modified";
      item.style.borderColor = "#e67e22";

      item.innerHTML = `
          <div class="pitch-number" style="background:#e67e22">${r.pitch_num}</div>
          <div class="pitch-info" style="flex:1;">
              <div class="pitch-result" style="color:#e67e22; margin-bottom: 8px;">
                  <strong>${r.type}</strong> (${r.zone})
              </div>
              <div class="badge-container">
                  <span class="badge badge-take" style="background:#7f8c8d;">👀 不揮棒 ${r.probs.take}%</span>
                  <span class="badge badge-whiff">💨 揮空 ${r.probs.whiff}%</span>
                  <span class="badge badge-foul">🛑 界外 ${r.probs.foul}%</span>
                  <span class="badge badge-weak">🏏 弱擊 ${r.probs.weak_contact}%</span>
                  <span class="badge badge-hard">🚀 強擊 ${r.probs.hard_contact}%</span>
              </div>
          </div>
      `;
      list.appendChild(item);
    });
  } catch (e) {
    console.error(e);
    alert("序列運算發生錯誤！請確認後端伺服器是否正常運作。");
  } finally {
    btn.innerText = "執行完整序列";
    btn.disabled = false;
  }
}

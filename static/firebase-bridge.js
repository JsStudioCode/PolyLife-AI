/**
 * PolyLife AI: Firebase Firestore Real-Time Bridge (Modular SDK v10)
 * Syncs assessments, TOPSIS weights, and Kaggle dataset logs in real time via onSnapshot.
 */

const FIREBASE_SDK_BASE = "https://www.gstatic.com/firebasejs/10.14.1";

class FirebaseBridge {
  constructor() {
    this.app = null;
    this.db = null;
    this.fs = null; // firestore module functions
    this.connected = false;
    this.config = null;
    this.unsubAssessments = null;
    this.unsubWeights = null;
    this.callbacks = {
      onStatus: () => {},
      onAssessments: () => {},
      onWeights: () => {},
    };
  }

  on(handlers) {
    Object.assign(this.callbacks, handlers);
  }

  async initFromServerOrStorage() {
    let cfg = null;
    try {
      const res = await fetch("/api/firebase/config");
      const data = await res.json();
      if (data && data.config) cfg = data.config;
    } catch (_) {}

    if (!cfg) {
      try {
        const raw = localStorage.getItem("polylife_firebase_config");
        if (raw) cfg = JSON.parse(raw);
      } catch (_) {}
    }

    if (cfg && cfg.apiKey && cfg.projectId) {
      try {
        await this.connect(cfg, { persistServer: false });
      } catch (err) {
        console.warn("Saved Firebase config failed to connect:", err);
        this.callbacks.onStatus(false, err.message);
      }
    } else {
      this.callbacks.onStatus(false, null);
    }
  }

  async connect(config, { persistServer = true } = {}) {
    if (!config || !config.apiKey || !config.projectId) {
      throw new Error("Firebase config requires 'apiKey' and 'projectId'.");
    }

    const [{ initializeApp, getApps, deleteApp }, firestoreMod] = await Promise.all([
      import(`${FIREBASE_SDK_BASE}/firebase-app.js`),
      import(`${FIREBASE_SDK_BASE}/firebase-firestore.js`),
    ]);

    for (const existing of getApps()) {
      await deleteApp(existing);
    }
    this.disconnectListeners();

    this.app = initializeApp(config, "polylife-ai");
    this.db = firestoreMod.getFirestore(this.app);
    this.fs = firestoreMod;
    this.config = config;

    // Verify connectivity with a lightweight read
    const qCheck = firestoreMod.query(
      firestoreMod.collection(this.db, "polylife_assessments"),
      firestoreMod.limit(1)
    );
    await firestoreMod.getDocs(qCheck);

    localStorage.setItem("polylife_firebase_config", JSON.stringify(config));
    if (persistServer) {
      await fetch("/api/firebase/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ config }),
      });
    }

    this.connected = true;
    this.startRealtimeListeners();
    this.callbacks.onStatus(true, config.projectId);
    return true;
  }

  disconnectListeners() {
    if (this.unsubAssessments) {
      this.unsubAssessments();
      this.unsubAssessments = null;
    }
    if (this.unsubWeights) {
      this.unsubWeights();
      this.unsubWeights = null;
    }
  }

  async disconnect() {
    this.disconnectListeners();
    this.connected = false;
    this.config = null;
    localStorage.removeItem("polylife_firebase_config");
    await fetch("/api/firebase/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ config: null }),
    }).catch(() => {});
    this.callbacks.onStatus(false, null);
  }

  startRealtimeListeners() {
    if (!this.connected || !this.fs) return;
    const { collection, query, orderBy, limit, onSnapshot, doc } = this.fs;

    const q = query(
      collection(this.db, "polylife_assessments"),
      orderBy("created_at", "desc"),
      limit(100)
    );
    this.unsubAssessments = onSnapshot(
      q,
      (snap) => {
        const items = [];
        snap.forEach((d) => items.push({ id: d.id, ...d.data() }));
        this.callbacks.onAssessments(items);
      },
      (err) => {
        console.warn("Firestore assessments listener error:", err);
      }
    );

    const wRef = doc(this.db, "polylife_weights", "active_profile");
    this.unsubWeights = onSnapshot(
      wRef,
      (snap) => {
        if (snap.exists() && Array.isArray(snap.data().weights)) {
          this.callbacks.onWeights(snap.data().weights);
        }
      },
      () => {}
    );
  }

  async pushAssessment(record) {
    if (!this.connected || !this.fs) return false;
    const { doc, setDoc } = this.fs;
    const ref = doc(this.db, "polylife_assessments", record.id);
    await setDoc(ref, record, { merge: true });
    return true;
  }

  async deleteAssessment(id) {
    if (!this.connected || !this.fs) return false;
    const { doc, deleteDoc } = this.fs;
    await deleteDoc(doc(this.db, "polylife_assessments", id));
    return true;
  }

  async pushWeights(weights) {
    if (!this.connected || !this.fs) return false;
    const { doc, setDoc } = this.fs;
    await setDoc(
      doc(this.db, "polylife_weights", "active_profile"),
      { weights, updated_at: Date.now() / 1000 },
      { merge: true }
    );
    return true;
  }

  async seedMany(records) {
    if (!this.connected || !this.fs) throw new Error("Firebase is not connected.");
    const { doc, writeBatch } = this.fs;
    const batch = writeBatch(this.db);
    for (const r of records.slice(0, 50)) {
      batch.set(doc(this.db, "polylife_assessments", r.id), r, { merge: true });
    }
    await batch.commit();
    return Math.min(records.length, 50);
  }
}

export const firebaseBridge = new FirebaseBridge();

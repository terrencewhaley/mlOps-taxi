import { useState, useEffect } from "react";

const API_BASE = "http://a6be6a0d4fe1d40dd86a9c653a5d7059-1557394960.us-east-1.elb.amazonaws.com";
const MLFLOW_BASE = "http://127.0.0.1:5001";

const NEIGHBORHOODS = {
  "JFK Airport": { lat: 40.6413, lng: -73.7781 },
  "LaGuardia Airport": { lat: 40.7769, lng: -73.8740 },
  "Midtown Manhattan": { lat: 40.7549, lng: -73.9840 },
  "Times Square": { lat: 40.7580, lng: -73.9855 },
  "Upper East Side": { lat: 40.7736, lng: -73.9566 },
  "Upper West Side": { lat: 40.7870, lng: -73.9754 },
  "Lower Manhattan": { lat: 40.7074, lng: -74.0113 },
  "Brooklyn Bridge": { lat: 40.7061, lng: -73.9969 },
  "Williamsburg": { lat: 40.7081, lng: -73.9571 },
  "Harlem": { lat: 40.8116, lng: -73.9465 },
  "Greenwich Village": { lat: 40.7336, lng: -74.0027 },
  "Grand Central": { lat: 40.7527, lng: -73.9772 },
};

const HOURS = Array.from({ length: 24 }, (_, i) => ({
  value: i,
  label: i === 0 ? "12:00 AM" : i < 12 ? `${i}:00 AM` : i === 12 ? "12:00 PM" : `${i - 12}:00 PM`,
}));

const STACK = [
  { label: "Model Training", tech: "scikit-learn + HistGradientBoosting" },
  { label: "Experiment Tracking", tech: "MLflow + SQLite" },
  { label: "API Layer", tech: "FastAPI + Pydantic" },
  { label: "Containerization", tech: "Docker (linux/amd64)" },
  { label: "Orchestration", tech: "Kubernetes on AWS EKS" },
  { label: "Container Registry", tech: "AWS ECR" },
  { label: "Model Storage", tech: "AWS S3 + IRSA Auth" },
  { label: "Infrastructure as Code", tech: "Terraform" },
  { label: "CI/CD", tech: "GitHub Actions" },
  { label: "Monitoring", tech: "Prometheus + Grafana" },
];

function StatusDot({ status }) {
  return (
    <span className="relative flex h-2 w-2">
      {status === "live" && (
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
      )}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${
        status === "live" ? "bg-emerald-400" :
        status === "checking" ? "bg-yellow-400" :
        "bg-red-500"
      }`} />
    </span>
  );
}

function Card({ children, className = "" }) {
  return (
    <div className={`bg-zinc-900 border border-zinc-800 rounded-lg p-5 ${className}`}>
      {children}
    </div>
  );
}

function SectionLabel({ children }) {
  return (
    <p className="text-xs font-mono text-zinc-500 uppercase tracking-widest mb-4">{children}</p>
  );
}

function Select({ value, onChange, options, className = "" }) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-xs text-zinc-100 font-mono focus:outline-none focus:border-emerald-500 transition-colors w-full ${className}`}
    >
      {options.map((o) => (
        <option key={o.value ?? o} value={o.value ?? o}>{o.label ?? o}</option>
      ))}
    </select>
  );
}

export default function PortfolioDashboard() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [activeTab, setActiveTab] = useState("predict");
  const [mlflowRuns, setMlflowRuns] = useState([]);
  const [mlflowError, setMlflowError] = useState(false);

  const [pickup, setPickup] = useState("Grand Central");
  const [dropoff, setDropoff] = useState("JFK Airport");
  const [hour, setHour] = useState(10);
  const [passengers, setPassengers] = useState(1);
  const [distance, setDistance] = useState(2.5);

  useEffect(() => {
    fetch(`${API_BASE}/health`)
      .then((r) => r.json())
      .then((d) => setApiStatus(d.status === "ok" ? "live" : "error"))
      .catch(() => setApiStatus("error"));

    fetch(`${MLFLOW_BASE}/api/2.0/mlflow/runs/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ experiment_ids: ["1"], max_results: 5, order_by: ["start_time DESC"] }),
    })
      .then((r) => r.json())
      .then((d) => setMlflowRuns(d.runs || []))
      .catch(() => setMlflowError(true));
  }, []);

  useEffect(() => {
    const p = NEIGHBORHOODS[pickup];
    const d = NEIGHBORHOODS[dropoff];
    if (p && d) {
      const R = 3958.8;
      const dLat = ((d.lat - p.lat) * Math.PI) / 180;
      const dLng = ((d.lng - p.lng) * Math.PI) / 180;
      const a = Math.sin(dLat / 2) ** 2 + Math.cos((p.lat * Math.PI) / 180) * Math.cos((d.lat * Math.PI) / 180) * Math.sin(dLng / 2) ** 2;
      const dist = R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
      setDistance(Math.round(dist * 10) / 10);
    }
  }, [pickup, dropoff]);

  const handlePredict = async () => {
    if (pickup === dropoff) return;
    setPredicting(true);
    setPrediction(null);
    const p = NEIGHBORHOODS[pickup];
    const d = NEIGHBORHOODS[dropoff];
    const datetime = `2016-01-15T${String(hour).padStart(2, "0")}:00:00`;
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tpep_pickup_datetime: datetime,
          Pickup_longitude: p.lng,
          Pickup_latitude: p.lat,
          Dropoff_longitude: d.lng,
          Dropoff_latitude: d.lat,
          Passenger_count: passengers,
          Trip_distance: distance,
        }),
      });
      const data = await res.json();
      setPrediction(data.predicted_fare);
    } catch {
      setPrediction("error");
    } finally {
      setPredicting(false);
    }
  };

  const neighborhoodOptions = Object.keys(NEIGHBORHOODS).map((n) => ({ value: n, label: n }));
  const passengerOptions = [1, 2, 3, 4, 5, 6].map((n) => ({ value: n, label: `${n} passenger${n > 1 ? "s" : ""}` }));

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100" style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />

      <div className="border-b border-zinc-800 px-8 py-5 flex items-center justify-between">
        <div>
          <h1 className="text-lg font-semibold tracking-tight" style={{ fontFamily: "'IBM Plex Sans', sans-serif" }}>
            NYC Taxi Fare — MLOps Pipeline
          </h1>
          <p className="text-xs text-zinc-500 mt-0.5">End-to-end ML system · AWS EKS · Kubernetes · Terraform</p>
        </div>
        <div className="flex items-center gap-2">
          <StatusDot status={apiStatus} />
          <span className="text-xs text-zinc-400 font-mono">
            {apiStatus === "live" ? "API live on AWS" : apiStatus === "checking" ? "checking..." : "API unreachable"}
          </span>
        </div>
      </div>

      <div className="px-8 py-8 max-w-6xl mx-auto space-y-6">

        <Card>
          <SectionLabel>Infrastructure Stack</SectionLabel>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
            {STACK.map((s) => (
              <div key={s.label} className="bg-zinc-800/50 rounded p-3 border border-zinc-700/50">
                <div className="flex items-center gap-1.5 mb-1">
                  <StatusDot status="live" />
                  <span className="text-xs text-zinc-300 font-medium">{s.label}</span>
                </div>
                <p className="text-xs text-zinc-500">{s.tech}</p>
              </div>
            ))}
          </div>
        </Card>

        <div className="flex gap-1 border-b border-zinc-800">
          {["predict", "mlflow", "architecture"].map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`px-4 py-2 text-xs font-mono uppercase tracking-wider transition-colors ${
                activeTab === tab
                  ? "text-emerald-400 border-b-2 border-emerald-400"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {activeTab === "predict" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <SectionLabel>Live Prediction — AWS EKS</SectionLabel>
              <div className="space-y-3">
                <div>
                  <label className="text-xs text-zinc-500 block mb-1.5">Pickup Location</label>
                  <Select value={pickup} onChange={setPickup} options={neighborhoodOptions} />
                </div>
                <div>
                  <label className="text-xs text-zinc-500 block mb-1.5">Dropoff Location</label>
                  <Select value={dropoff} onChange={setDropoff} options={neighborhoodOptions} />
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="text-xs text-zinc-500 block mb-1.5">Pickup Hour</label>
                    <Select value={hour} onChange={(v) => setHour(Number(v))} options={HOURS} />
                  </div>
                  <div>
                    <label className="text-xs text-zinc-500 block mb-1.5">Passengers</label>
                    <Select value={passengers} onChange={(v) => setPassengers(Number(v))} options={passengerOptions} />
                  </div>
                </div>
                <div className="bg-zinc-800/50 rounded p-3 border border-zinc-700/50">
                  <p className="text-xs text-zinc-500">Estimated distance</p>
                  <p className="text-sm text-zinc-300 font-mono mt-0.5">{distance} miles</p>
                  <p className="text-xs text-zinc-600 mt-1">Auto-calculated from coordinates</p>
                </div>
              </div>
              <button
                onClick={handlePredict}
                disabled={predicting || pickup === dropoff}
                className="mt-5 w-full bg-emerald-500 hover:bg-emerald-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-zinc-950 font-semibold text-xs py-2.5 rounded transition-colors"
              >
                {predicting ? "predicting..." : pickup === dropoff ? "select different locations" : "run prediction"}
              </button>
            </Card>

            <Card className="flex flex-col justify-between">
              <div>
                <SectionLabel>Prediction Result</SectionLabel>
                {prediction !== null ? (
                  prediction === "error" ? (
                    <p className="text-red-400 text-sm mt-8">Request failed — check API status</p>
                  ) : (
                    <div className="mt-6">
                      <div className="flex items-end gap-2">
                        <span className="text-6xl font-semibold text-emerald-400" style={{ fontFamily: "'IBM Plex Sans', sans-serif" }}>
                          ${prediction}
                        </span>
                        <span className="text-zinc-500 text-sm mb-2">predicted fare</span>
                      </div>
                      <p className="text-xs text-zinc-500 mt-3">{pickup} → {dropoff}</p>
                      <p className="text-xs text-zinc-600 mt-1">
                        {HOURS.find(h => h.value === hour)?.label} · {passengers} passenger{passengers > 1 ? "s" : ""} · {distance} mi
                      </p>
                    </div>
                  )
                ) : (
                  <p className="text-zinc-600 text-sm mt-8">Select locations and run a prediction</p>
                )}
              </div>
              <div className="border-t border-zinc-800 pt-4 mt-4 space-y-2">
                <SectionLabel>Endpoint</SectionLabel>
                <p className="text-xs text-zinc-500 font-mono break-all">{API_BASE}/predict</p>
                <p className="text-xs text-zinc-600">POST · JSON · AWS Load Balancer → EKS → FastAPI</p>
              </div>
            </Card>
          </div>
        )}

        {activeTab === "mlflow" && (
          <Card>
            <SectionLabel>MLflow Experiment Runs — nyc-taxi-fare-prediction</SectionLabel>
            {mlflowError ? (
              <div className="bg-zinc-800/50 rounded p-4 border border-zinc-700/50">
                <p className="text-xs text-zinc-400 mb-2">MLflow UI not running locally. Start it with:</p>
                <p className="text-xs text-emerald-400 font-mono break-all">mlflow ui --backend-store-uri sqlite:////Users/twhaley/Desktop/mlOps-taxi/mlflow.db --port 5001</p>
              </div>
            ) : mlflowRuns.length === 0 ? (
              <p className="text-xs text-zinc-500">Loading runs...</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-zinc-800">
                      {["Run ID", "RMSE", "MAE", "R²", "Learning Rate", "Max Depth", "Status"].map((h) => (
                        <th key={h} className="text-left text-zinc-500 pb-2 pr-6 font-normal">{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {mlflowRuns.map((run) => {
                      const m = run.data?.metrics || {};
                      const p = run.data?.params || {};
                      return (
                        <tr key={run.info?.run_id} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
                          <td className="py-2.5 pr-6 text-zinc-400">{run.info?.run_id?.slice(0, 8)}...</td>
                          <td className="py-2.5 pr-6 text-emerald-400">{m.rmse?.toFixed(4) ?? "—"}</td>
                          <td className="py-2.5 pr-6 text-emerald-400">{m.mae?.toFixed(4) ?? "—"}</td>
                          <td className="py-2.5 pr-6 text-emerald-400">{m.r2?.toFixed(4) ?? "—"}</td>
                          <td className="py-2.5 pr-6 text-zinc-300">{p.learning_rate ?? "—"}</td>
                          <td className="py-2.5 pr-6 text-zinc-300">{p.max_depth ?? "—"}</td>
                          <td className="py-2.5">
                            <span className={`px-2 py-0.5 rounded text-xs ${run.info?.status === "FINISHED" ? "bg-emerald-900/50 text-emerald-400" : "bg-zinc-800 text-zinc-400"}`}>
                              {run.info?.status ?? "—"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        )}

        {activeTab === "architecture" && (
          <div className="space-y-4">
            <Card>
              <SectionLabel>Data Flow</SectionLabel>
              <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
                {["NYC Taxi CSV", "→", "train.py", "→", "MLflow Tracking", "→", "model.joblib", "→", "AWS S3", "→", "FastAPI Container", "→", "EKS Pod", "→", "Load Balancer", "→", "Client"].map((step, i) => (
                  <span key={i} className={step === "→" ? "text-zinc-600" : "bg-zinc-800 px-2 py-1 rounded text-zinc-300"}>
                    {step}
                  </span>
                ))}
              </div>
            </Card>

            <Card>
              <SectionLabel>Deployment Flow</SectionLabel>
              <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
                {["git push", "→", "GitHub Actions", "→", "Docker Build (amd64)", "→", "AWS ECR", "→", "kubectl rollout", "→", "EKS (2 replicas)"].map((step, i) => (
                  <span key={i} className={step === "→" ? "text-zinc-600" : "bg-zinc-800 px-2 py-1 rounded text-zinc-300"}>
                    {step}
                  </span>
                ))}
              </div>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <SectionLabel>Model Performance</SectionLabel>
                <p className="text-xs text-zinc-300 mb-3">HistGradientBoostingRegressor</p>
                <div className="space-y-2">
                  {[["RMSE", "3.30"], ["MAE", "1.42"], ["R²", "0.9009"]].map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-xs text-zinc-500">{k}</span>
                      <span className="text-xs text-emerald-400 font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <SectionLabel>Infrastructure</SectionLabel>
                <div className="space-y-2">
                  {[["Region", "us-east-1"], ["Nodes", "2x t3.small"], ["Replicas", "2"], ["Auth", "IRSA"]].map(([k, v]) => (
                    <div key={k} className="flex justify-between">
                      <span className="text-xs text-zinc-500">{k}</span>
                      <span className="text-xs text-zinc-300 font-mono">{v}</span>
                    </div>
                  ))}
                </div>
              </Card>
              <Card>
                <SectionLabel>Repository</SectionLabel>
                <a
                  href="https://github.com/terrencewhaley/mlOps-taxi"
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors block mb-3"
                >
                  github.com/terrencewhaley/mlOps-taxi →
                </a>
                <p className="text-xs text-zinc-500 leading-relaxed">FastAPI · Docker · Kubernetes · Terraform · MLflow · Prometheus · Grafana</p>
              </Card>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

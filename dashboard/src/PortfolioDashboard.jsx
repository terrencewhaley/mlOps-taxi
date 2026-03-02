import { useState, useEffect } from "react";

const API_BASE = "http://a6be6a0d4fe1d40dd86a9c653a5d7059-1557394960.us-east-1.elb.amazonaws.com";
const MLFLOW_BASE = "http://127.0.0.1:5001";

const STACK = [
  { label: "Model Training", tech: "scikit-learn + HistGradientBoosting", status: "live" },
  { label: "Experiment Tracking", tech: "MLflow + SQLite", status: "live" },
  { label: "API Layer", tech: "FastAPI + Pydantic", status: "live" },
  { label: "Containerization", tech: "Docker (linux/amd64)", status: "live" },
  { label: "Orchestration", tech: "Kubernetes on AWS EKS", status: "live" },
  { label: "Container Registry", tech: "AWS ECR", status: "live" },
  { label: "Model Storage", tech: "AWS S3 + IRSA Auth", status: "live" },
  { label: "Infrastructure as Code", tech: "Terraform", status: "live" },
  { label: "CI/CD", tech: "GitHub Actions", status: "live" },
  { label: "Monitoring", tech: "Prometheus + Grafana", status: "live" },
];

const DEFAULT_TRIP = {
  tpep_pickup_datetime: "2016-01-01T10:00:00",
  Pickup_longitude: -73.9857,
  Pickup_latitude: 40.7484,
  Dropoff_longitude: -73.9654,
  Dropoff_latitude: 40.7829,
  Passenger_count: 1,
  Trip_distance: 2.5,
};

function StatusDot({ status }) {
  return (
    <span className="relative flex h-2 w-2">
      {status === "live" && (
        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
      )}
      <span className={`relative inline-flex rounded-full h-2 w-2 ${status === "live" ? "bg-emerald-400" : status === "checking" ? "bg-yellow-400" : "bg-red-500"}`} />
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

export default function PortfolioDashboard() {
  const [apiStatus, setApiStatus] = useState("checking");
  const [prediction, setPrediction] = useState(null);
  const [predicting, setPredicting] = useState(false);
  const [trip, setTrip] = useState(DEFAULT_TRIP);
  const [mlflowRuns, setMlflowRuns] = useState([]);
  const [mlflowError, setMlflowError] = useState(false);
  const [activeTab, setActiveTab] = useState("predict");

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

  const handlePredict = async () => {
    setPredicting(true);
    setPrediction(null);
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(trip),
      });
      const data = await res.json();
      setPrediction(data.predicted_fare);
    } catch {
      setPrediction("error");
    } finally {
      setPredicting(false);
    }
  };

  const handleChange = (key, value) => {
    setTrip((prev) => ({ ...prev, [key]: isNaN(value) ? value : Number(value) }));
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-100" style={{ fontFamily: "'IBM Plex Mono', monospace" }}>
      <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600&display=swap" rel="stylesheet" />

      {/* Header */}
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

        {/* Stack overview */}
        <Card>
          <SectionLabel>Infrastructure Stack</SectionLabel>
          <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
            {STACK.map((s) => (
              <div key={s.label} className="bg-zinc-800/50 rounded p-3 border border-zinc-700/50">
                <div className="flex items-center gap-1.5 mb-1">
                  <StatusDot status={s.status} />
                  <span className="text-xs text-zinc-300 font-medium">{s.label}</span>
                </div>
                <p className="text-xs text-zinc-500">{s.tech}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Tabs */}
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

        {/* Predict Tab */}
        {activeTab === "predict" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <SectionLabel>Live Prediction — AWS EKS</SectionLabel>
              <div className="space-y-3">
                {[
                  { key: "tpep_pickup_datetime", label: "Pickup Time", type: "text" },
                  { key: "Pickup_longitude", label: "Pickup Longitude", type: "number" },
                  { key: "Pickup_latitude", label: "Pickup Latitude", type: "number" },
                  { key: "Dropoff_longitude", label: "Dropoff Longitude", type: "number" },
                  { key: "Dropoff_latitude", label: "Dropoff Latitude", type: "number" },
                  { key: "Passenger_count", label: "Passengers", type: "number" },
                  { key: "Trip_distance", label: "Distance (miles)", type: "number" },
                ].map(({ key, label, type }) => (
                  <div key={key} className="flex items-center justify-between gap-4">
                    <label className="text-xs text-zinc-500 w-36 shrink-0">{label}</label>
                    <input
                      type={type}
                      value={trip[key]}
                      onChange={(e) => handleChange(key, e.target.value)}
                      className="flex-1 bg-zinc-800 border border-zinc-700 rounded px-3 py-1.5 text-xs text-zinc-100 font-mono focus:outline-none focus:border-emerald-500 transition-colors"
                    />
                  </div>
                ))}
              </div>
              <button
                onClick={handlePredict}
                disabled={predicting}
                className="mt-5 w-full bg-emerald-500 hover:bg-emerald-400 disabled:bg-zinc-700 disabled:text-zinc-500 text-zinc-950 font-semibold text-xs py-2.5 rounded transition-colors"
              >
                {predicting ? "predicting..." : "run prediction"}
              </button>
            </Card>

            <Card className="flex flex-col justify-between">
              <div>
                <SectionLabel>Prediction Result</SectionLabel>
                <div className="flex items-end gap-2 mt-8 mb-2">
                  {prediction !== null ? (
                    prediction === "error" ? (
                      <p className="text-red-400 text-sm">Request failed — check API status</p>
                    ) : (
                      <>
                        <span className="text-6xl font-semibold text-emerald-400" style={{ fontFamily: "'IBM Plex Sans', sans-serif" }}>
                          ${prediction}
                        </span>
                        <span className="text-zinc-500 text-sm mb-2">predicted fare</span>
                      </>
                    )
                  ) : (
                    <p className="text-zinc-600 text-sm">Submit a trip to see the prediction</p>
                  )}
                </div>
              </div>
              <div className="border-t border-zinc-800 pt-4 mt-4 space-y-2">
                <SectionLabel>Endpoint</SectionLabel>
                <p className="text-xs text-zinc-500 font-mono break-all">{API_BASE}/predict</p>
                <p className="text-xs text-zinc-600">POST · JSON · AWS Load Balancer → EKS → FastAPI</p>
              </div>
            </Card>
          </div>
        )}

        {/* MLflow Tab */}
        {activeTab === "mlflow" && (
          <Card>
            <SectionLabel>MLflow Experiment Runs — nyc-taxi-fare-prediction</SectionLabel>
            {mlflowError ? (
              <p className="text-xs text-zinc-500">MLflow UI not running locally. Start it with: <span className="text-emerald-400">mlflow ui --backend-store-uri sqlite:////Users/twhaley/Desktop/mlOps-taxi/mlflow.db --port 5001</span></p>
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
                          <td className="py-2.5 pr-6 text-zinc-400 font-mono">{run.info?.run_id?.slice(0, 8)}...</td>
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

        {/* Architecture Tab */}
        {activeTab === "architecture" && (
          <div className="space-y-4">
            <Card>
              <SectionLabel>Data Flow</SectionLabel>
              <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
                {[
                  "NYC Taxi CSV",
                  "→",
                  "train.py",
                  "→",
                  "MLflow Tracking",
                  "→",
                  "model.joblib",
                  "→",
                  "AWS S3",
                  "→",
                  "FastAPI Container",
                  "→",
                  "EKS Pod",
                  "→",
                  "Load Balancer",
                  "→",
                  "Client",
                ].map((step, i) => (
                  <span key={i} className={step === "→" ? "text-zinc-600" : "bg-zinc-800 px-2 py-1 rounded text-zinc-300"}>
                    {step}
                  </span>
                ))}
              </div>
            </Card>

            <Card>
              <SectionLabel>Deployment Flow</SectionLabel>
              <div className="flex items-center gap-2 flex-wrap text-xs font-mono">
                {[
                  "git push",
                  "→",
                  "GitHub Actions",
                  "→",
                  "Docker Build (amd64)",
                  "→",
                  "AWS ECR",
                  "→",
                  "kubectl rollout",
                  "→",
                  "EKS (2 replicas)",
                ].map((step, i) => (
                  <span key={i} className={step === "→" ? "text-zinc-600" : "bg-zinc-800 px-2 py-1 rounded text-zinc-300"}>
                    {step}
                  </span>
                ))}
              </div>
            </Card>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <Card>
                <SectionLabel>Model</SectionLabel>
                <p className="text-xs text-zinc-300">HistGradientBoostingRegressor</p>
                <div className="mt-3 space-y-1">
                  <p className="text-xs text-zinc-500">RMSE <span className="text-emerald-400 ml-2">3.30</span></p>
                  <p className="text-xs text-zinc-500">MAE <span className="text-emerald-400 ml-2">1.42</span></p>
                  <p className="text-xs text-zinc-500">R² <span className="text-emerald-400 ml-2">0.9009</span></p>
                </div>
              </Card>
              <Card>
                <SectionLabel>Infrastructure</SectionLabel>
                <div className="space-y-1">
                  <p className="text-xs text-zinc-500">Region <span className="text-zinc-300 ml-2">us-east-1</span></p>
                  <p className="text-xs text-zinc-500">Nodes <span className="text-zinc-300 ml-2">2x t3.small</span></p>
                  <p className="text-xs text-zinc-500">Replicas <span className="text-zinc-300 ml-2">2</span></p>
                  <p className="text-xs text-zinc-500">Auth <span className="text-zinc-300 ml-2">IRSA</span></p>
                </div>
              </Card>
              <Card>
                <SectionLabel>Repository</SectionLabel>
                <a
                  href="https://github.com/terrencewhaley/mlOps-taxi"
                  target="_blank"
                  rel="noreferrer"
                  className="text-xs text-emerald-400 hover:text-emerald-300 transition-colors"
                >
                  github.com/terrencewhaley/mlOps-taxi →
                </a>
                <p className="text-xs text-zinc-500 mt-2">FastAPI · Docker · K8s · Terraform · MLflow · Prometheus</p>
              </Card>
            </div>
          </div>
        )}

      </div>
    </div>
  );
}

import { useEffect, useState } from "react";
import { getDriftStatus } from "../api";

export default function DriftAlert() {
  const [drift, setDrift] = useState(false);

  useEffect(() => {
    getDriftStatus().then(res => {
      setDrift(res.data.drift);
    });
  }, []);

  if (!drift) return null;
  return (
    <div className="p-3 bg-yellow-200 text-yellow-900 rounded">
      ⚠️ Data drift detected! Model may be degraded.
    </div>
  );
}

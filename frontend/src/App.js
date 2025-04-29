import { useState } from "react";
import Upload from "./components/Upload";
import Result from "./components/Result";
import DriftAlert from "./components/DriftAlert";
import Feedback from "./components/Feedback";

function App() {
  const [result, setResult] = useState(null);

  return (
    <div className="max-w-lg mx-auto p-6">
      <h1 className="text-2xl font-bold mb-4">
        ALL Classifier
      </h1>
      <DriftAlert />
      <Upload onResult={setResult} />
      <Result result={result} />
      {result?.image_id && <Feedback imageId={result.image_id} />}
    </div>
  );
}

export default App;

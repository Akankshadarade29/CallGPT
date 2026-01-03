import { useState } from "react";
import "./App.css";

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  return (
    <div className="app">
      <h2 style={{ padding: "10px", background: "#202123", color: "white" }}>
        💬 CallGPT
      </h2>

      <div style={{ padding: "20px" }}>
        <p>This is CallGPT frontend</p>
      </div>

      <div style={{ padding: "10px" }}>
        <input
          placeholder="Type message..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
        />
      </div>
    </div>
  );
}

export default App;

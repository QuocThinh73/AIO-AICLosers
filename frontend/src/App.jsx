import { useState } from "react";
import SearchBar from "./components/SearchBar";
import baseSearch from "./services/searchApi";
import ResultList from "./components/ResultList";


function App() {
  const [keyframes, setKeyframes] = useState([]);

  async function handleSearch(params) {
    setKeyframes([]);
    try {
      const results = await baseSearch(params);
      setKeyframes(Array.isArray(results) ? results : Array.from(results ?? []));
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div>
      <SearchBar onSearch={handleSearch} />
      <ResultList keyframes={keyframes} />
    </div>
  )
}

export default App;
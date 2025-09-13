import { useState } from "react";
import SearchBar from "./components/SearchBar";
import baseSearch from "./services/searchApi";
import ResultContainer from "./components/ResultContainer";

function App() {
  const [keyframes, setKeyframes] = useState([]);
  const [englishQuery, setEnglishQuery] = useState("");

  async function handleSearch(params) {
    setKeyframes([]);
    try {
      const { results, english_query } = await baseSearch(params);
      setKeyframes(
        Array.isArray(results) ? results : Array.from(results ?? [])
      );
      setEnglishQuery(english_query);
    } catch (error) {
      console.error(error);
    }
  }

  return (
    <div>
      <SearchBar onSearch={handleSearch} englishQuery={englishQuery} />
      <ResultContainer keyframes={keyframes} />
    </div>
  );
}

export default App;

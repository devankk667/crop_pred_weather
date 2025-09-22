document.getElementById("predictForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData(e.target);
  const data = Object.fromEntries(formData.entries());

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data)
    });

    if (!res.ok) throw new Error((await res.json()).detail || "Error");

    const json = await res.json();
    const summary = json.weather_used.summary;
    const daily = json.weather_used.daily;

    document.getElementById("result").innerHTML = `
      ✅ Predicted Yield: <strong>${json.predicted_yield}</strong> tons/ha<br/>
      🌾 Crop: ${data.crop} · ${data.season} ${data.year}<br/>
      ⚡ Model v${json.model_version}<br/>
      🌤️ Weather Summary:<br/>
      Avg Temp: ${summary.avg_temp.toFixed(1)}°C, 
      Total Precip: ${summary.total_precip.toFixed(1)}mm, 
      Avg Humidity: ${summary.avg_humidity.toFixed(1)}%, 
      Avg Wind Speed: ${summary.avg_windspeed.toFixed(1)} m/s
    `;
    document.getElementById("result").classList.remove("hidden");

  } catch (err) {
    document.getElementById("result").textContent = "Error: " + err.message;
    document.getElementById("result").classList.remove("hidden");
  }
});

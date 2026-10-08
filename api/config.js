// Vercel serverless function: hands the deployment's environment to the static
// frontend. public/script.js fetches /api/config at runtime and uses the values
// as the backend URL (and optional API key).
//
// Set these in Vercel -> Project -> Settings -> Environment Variables:
//   BACKEND_URL   e.g. https://rndf-api.onrender.com   (no trailing slash needed)
//   DEMO_API_KEY  must equal the Render env API_KEY (leave both unset to disable auth)
//
// Note: anything served to a browser is public by definition - treat DEMO_API_KEY
// as light gating, not a secret.
module.exports = (req, res) => {
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");
  res.setHeader("Cache-Control", "no-store");
  res.statusCode = 200;
  res.setHeader("Content-Type", "application/json");
  res.end(
    JSON.stringify({
      backendUrl: process.env.BACKEND_URL || "",
      apiKey: process.env.DEMO_API_KEY || "",
    })
  );
};

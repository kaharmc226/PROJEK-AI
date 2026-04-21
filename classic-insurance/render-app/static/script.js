/**
 * Insurance Cost Predictor — Client-side Logic
 * Handles form submission, API calls, and result rendering.
 */

(function () {
    "use strict";

    // DOM refs
    const form         = document.getElementById("predict-form");
    const submitBtn    = document.getElementById("submit-btn");
    const resultCard   = document.getElementById("result-card");
    const resultAmount = document.getElementById("result-amount");
    const resultSummary= document.getElementById("result-input-summary");
    const warningsSec  = document.getElementById("warnings-section");
    const warningsList = document.getElementById("warnings-list");
    const featureTbody = document.getElementById("feature-tbody");
    const smokerHidden = document.getElementById("smoker");

    // ---- Smoker toggle ----
    document.querySelectorAll(".toggle-btn").forEach((btn) => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".toggle-btn").forEach((b) => b.classList.remove("active"));
            btn.classList.add("active");
            smokerHidden.value = btn.dataset.value;
        });
    });

    // ---- Form submit ----
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        // Collect values
        const payload = {
            age:      parseInt(document.getElementById("age").value, 10),
            sex:      document.getElementById("sex").value,
            bmi:      parseFloat(document.getElementById("bmi").value),
            children: parseInt(document.getElementById("children").value, 10),
            smoker:   smokerHidden.value,
            region:   document.getElementById("region").value,
        };

        // Basic client-side validation
        if (isNaN(payload.age) || isNaN(payload.bmi) || isNaN(payload.children)) {
            alert("Please fill in all numeric fields.");
            return;
        }

        // Show loading state
        submitBtn.classList.add("loading");
        submitBtn.disabled = true;

        try {
            const resp = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.error || `Server error (${resp.status})`);
            }

            const data = await resp.json();
            renderResult(data, payload);

        } catch (err) {
            alert("Prediction failed: " + err.message);
        } finally {
            submitBtn.classList.remove("loading");
            submitBtn.disabled = false;
        }
    });

    // ---- Render result ----
    function renderResult(data, input) {
        // Show card
        resultCard.classList.remove("hidden");
        resultCard.scrollIntoView({ behavior: "smooth", block: "start" });

        // Animated counter for the dollar amount
        animateCounter(resultAmount, data.predicted_charges);

        // Input summary
        resultSummary.innerHTML =
            `${capitalize(input.sex)}, Age ${input.age}, BMI ${input.bmi}, ` +
            `${input.children} child${input.children !== 1 ? "ren" : ""}, ` +
            `${input.smoker === "yes" ? "Smoker" : "Non-smoker"}, ` +
            `${capitalize(input.region)}`;

        // Warnings
        if (data.warnings && data.warnings.length > 0) {
            warningsSec.classList.remove("hidden");
            warningsList.innerHTML = data.warnings.map((w) => `<li>${w}</li>`).join("");
        } else {
            warningsSec.classList.add("hidden");
        }

        // Feature table
        featureTbody.innerHTML = "";
        for (const [name, value] of Object.entries(data.features)) {
            const tr = document.createElement("tr");
            tr.innerHTML = `<td>${name}</td><td>${formatValue(value)}</td>`;
            featureTbody.appendChild(tr);
        }
    }

    // ---- Helpers ----
    function capitalize(s) {
        return s.charAt(0).toUpperCase() + s.slice(1);
    }

    function formatValue(v) {
        if (Number.isInteger(v)) return v.toString();
        return v.toFixed(4);
    }

    function animateCounter(el, target) {
        const duration = 800; // ms
        const start = performance.now();
        const startVal = 0;

        function step(now) {
            const progress = Math.min((now - start) / duration, 1);
            // ease-out cubic
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = startVal + (target - startVal) * eased;
            el.textContent = "$" + current.toLocaleString("en-US", {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
            });
            if (progress < 1) requestAnimationFrame(step);
        }

        requestAnimationFrame(step);
    }
})();

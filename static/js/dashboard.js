// Toggle between User Details and Policies
const userDetailsBtn = document.getElementById("user-details-btn");
const policiesBtn = document.getElementById("policies-btn");
const userDetailsSection = document.getElementById("user-details");
const policiesSection = document.getElementById("policies");

// Initially hide the policies section
policiesSection.style.display = "none";

userDetailsBtn.addEventListener("click", () => {
    userDetailsBtn.classList.add("active");
    policiesBtn.classList.remove("active");
    userDetailsSection.style.display = "block";
    policiesSection.style.display = "none";
});

policiesBtn.addEventListener("click", () => {
    policiesBtn.classList.add("active");
    userDetailsBtn.classList.remove("active");
    policiesSection.style.display = "block";
    userDetailsSection.style.display = "none";
});

// Nested Toggles for Policies
const currentPoliciesBtn = document.getElementById("current-policies-btn");
const completedPoliciesBtn = document.getElementById("completed-policies-btn");
const currentPoliciesSection = document.getElementById("current-policies-section");
const completedPoliciesSection = document.getElementById("completed-policies-section");

// Initially hide the completed policies section
completedPoliciesSection.style.display = "none";

currentPoliciesBtn.addEventListener("click", () => {
    currentPoliciesBtn.classList.add("active");
    completedPoliciesBtn.classList.remove("active");
    currentPoliciesSection.style.display = "block";
    completedPoliciesSection.style.display = "none";
});

completedPoliciesBtn.addEventListener("click", () => {
    completedPoliciesBtn.classList.add("active");
    currentPoliciesBtn.classList.remove("active");
    completedPoliciesSection.style.display = "block";
    currentPoliciesSection.style.display = "none";
});

// Function to toggle policy details
function toggleDetails(button) {
    const detailsArea = button.closest(".policy-box").querySelector(".policy-details-area");
    const logo = detailsArea.querySelector(".policy-logo");
    const details = detailsArea.querySelector(".policy-details");

    logo.classList.toggle("hidden");
    details.classList.toggle("hidden");
}
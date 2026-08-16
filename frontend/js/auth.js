const tabSignin = document.getElementById("tab-signin");
const tabSignup = document.getElementById("tab-signup");
const signinForm = document.getElementById("signin-form");
const signupForm = document.getElementById("signup-form");
const signinError = document.getElementById("signin-error");
const signupError = document.getElementById("signup-error");

function showTab(tab) {
  const isSignin = tab === "signin";
  tabSignin.classList.toggle("active", isSignin);
  tabSignup.classList.toggle("active", !isSignin);
  signinForm.style.display = isSignin ? "flex" : "none";
  signupForm.style.display = isSignin ? "none" : "flex";
}

tabSignin.addEventListener("click", () => showTab("signin"));
tabSignup.addEventListener("click", () => showTab("signup"));

function redirectForRole(role) {
  window.location.href = role === "admin" ? "/admin.html" : "/recruiter.html";
}

function showError(el, message) {
  el.textContent = message;
  el.style.display = "block";
}

function hideError(el) {
  el.style.display = "none";
}

signinForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError(signinError);
  const formData = new FormData(signinForm);
  const submitBtn = signinForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  try {
    const user = await api.signin(formData.get("email"), formData.get("password"));
    redirectForRole(user.role);
  } catch (err) {
    showError(signinError, err.message || "Sign in failed. Please try again.");
  } finally {
    submitBtn.disabled = false;
  }
});

signupForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError(signupError);
  const formData = new FormData(signupForm);
  const payload = Object.fromEntries(formData.entries());
  const submitBtn = signupForm.querySelector('button[type="submit"]');
  submitBtn.disabled = true;
  try {
    const user = await api.signup(payload);
    redirectForRole(user.role);
  } catch (err) {
    showError(signupError, err.message || "Sign up failed. Please try again.");
  } finally {
    submitBtn.disabled = false;
  }
});

// Already signed in? Skip the form entirely rather than showing a login page to someone
// who's already authenticated.
api
  .getMe()
  .then((user) => redirectForRole(user.role))
  .catch(() => {
    // Not signed in — this is the expected case, stay on the auth page.
  });

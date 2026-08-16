(function () {
  const toggle = document.getElementById("nav-toggle");
  const nav = document.querySelector(".top-nav");
  if (!toggle || !nav) return;
  toggle.addEventListener("click", () => {
    nav.classList.toggle("open");
  });
})();

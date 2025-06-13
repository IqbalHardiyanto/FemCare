// Hamburger Menu Toggle
const hamburger = document.getElementById("hamburger");
const mobileMenu = document.getElementById("mobile-menu");

hamburger.addEventListener("click", () => {
  mobileMenu.classList.toggle("hidden");
  mobileMenu.classList.toggle("animate-slideDown");

  // Ganti icon hamburger <-> close
  const icon = hamburger.querySelector("i");
  if (icon.classList.contains("fa-bars")) {
    icon.classList.remove("fa-bars");
    icon.classList.add("fa-times");
  } else {
    icon.classList.remove("fa-times");
    icon.classList.add("fa-bars");
  }
});

// Navbar Scroll Effect
window.addEventListener("scroll", () => {
  const navbar = document.getElementById("navbar");
  if (window.scrollY > 10) {
    navbar.classList.add("shadow-lg");
    navbar.classList.add("bg-blue-700");
    navbar.classList.remove("bg-blue-600");
  } else {
    navbar.classList.remove("shadow-lg");
    navbar.classList.remove("bg-blue-700");
    navbar.classList.add("bg-blue-600");
  }
});

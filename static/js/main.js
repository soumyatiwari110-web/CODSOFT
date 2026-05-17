/* ============================================================
   TITANIC SURVIVAL PREDICTION — JavaScript
   Author: Soumya Tiwari
   ============================================================ */

document.addEventListener('DOMContentLoaded', function () {

  // ---- AGE SLIDER SYNC ----
  const ageSlider = document.getElementById('ageSlider');
  const ageInput  = document.querySelector('input[name="age"]');
  const ageVal    = document.getElementById('ageVal');

  if (ageSlider && ageInput) {
    ageSlider.addEventListener('input', function () {
      ageInput.value = this.value;
      if (ageVal) ageVal.textContent = this.value;
    });

    ageInput.addEventListener('input', function () {
      if (this.value >= 1 && this.value <= 80) {
        ageSlider.value = this.value;
        if (ageVal) ageVal.textContent = this.value;
      }
    });
  }

  // ---- FORM SUBMIT ANIMATION ----
  const form = document.getElementById('predictionForm');
  if (form) {
    form.addEventListener('submit', function () {
      const btnText    = document.getElementById('btnText');
      const btnLoading = document.getElementById('btnLoading');
      if (btnText && btnLoading) {
        btnText.style.display    = 'none';
        btnLoading.style.display = 'inline';
      }
    });
  }

  // ---- PROBABILITY BAR ANIMATION ----
  const bars = document.querySelectorAll('.st-prob-bar');
  bars.forEach(bar => {
    const targetWidth = bar.getAttribute('data-width');
    bar.style.width = '0%';
    setTimeout(() => {
      bar.style.width = targetWidth + '%';
    }, 400);
  });

  // ---- NAVBAR SCROLL EFFECT ----
  const navbar = document.querySelector('.st-navbar');
  if (navbar) {
    window.addEventListener('scroll', function () {
      if (window.scrollY > 50) {
        navbar.style.background = 'rgba(13,27,42,0.98)';
        navbar.style.boxShadow  = '0 4px 20px rgba(0,0,0,0.4)';
      } else {
        navbar.style.background = 'rgba(13,27,42,0.95)';
        navbar.style.boxShadow  = 'none';
      }
    });
  }

  // ---- SCROLL REVEAL ----
  const revealEls = document.querySelectorAll(
    '.st-card, .st-insight-card, .st-about-card'
  );

  const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.style.opacity   = '1';
        entry.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.15 });

  revealEls.forEach(el => {
    el.style.opacity    = '0';
    el.style.transform  = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
    observer.observe(el);
  });

});

// ---- COUNTER BUTTONS ----
function changeCount(fieldId, delta) {
  const input = document.getElementById(fieldId);
  if (!input) return;
  const maxVals = { sibsp: 8, parch: 6 };
  let val = parseInt(input.value) + delta;
  val = Math.max(0, Math.min(val, maxVals[fieldId] || 10));
  input.value = val;
}
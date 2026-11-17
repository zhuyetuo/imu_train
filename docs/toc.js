(function () {
  var hs = document.querySelectorAll(".wrap h2");
  if (hs.length < 3) return;
  var toc = document.createElement("div");
  toc.id = "toc";
  hs.forEach(function (h, i) {
    if (!h.id) h.id = "s" + (i + 1);
    var a = document.createElement("a");
    a.href = "#" + h.id;
    a.textContent = h.textContent;
    toc.appendChild(a);
  });
  document.body.appendChild(toc);
  var links = toc.querySelectorAll("a");
  function onScroll() {
    var cur = 0;
    hs.forEach(function (h, i) { if (h.getBoundingClientRect().top < 120) cur = i; });
    links.forEach(function (a, i) { a.classList.toggle("on", i === cur); });
  }
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
})();

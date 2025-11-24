window.HELP_IMPROVE_VIDEOJS = false;

$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    // ------------------------------------------------------------
    // Lazy-load Rerun iframes when drawers are opened
    // ------------------------------------------------------------
    const drawers = document.querySelectorAll(".rerun-drawer");

    drawers.forEach(drawer => {
        drawer.addEventListener("toggle", () => {
            if (drawer.open) {
                const container = drawer.querySelector(".iframe-container");
                if (container && !container.querySelector("iframe")) {
                    const iframe = document.createElement("iframe");
                    iframe.src = container.dataset.src;
                    iframe.style.width = "100%";
                    iframe.style.height = "500px";
                    iframe.style.border = "none";
                    iframe.loading = "lazy";
                    container.appendChild(iframe);
                }
            }
        });
    });


    // ------------------------------------------------------------
    // Image modal popup (for figure enlarging)
    // ------------------------------------------------------------
    const modal = document.getElementById("image-modal");
    const modalImg = modal ? modal.querySelector("img") : null;

    // Attach to any image with data-enlarge attribute or a specific ID
    const clickableImages = document.querySelectorAll("#method-overview-img, [data-enlarge]");

    clickableImages.forEach(img => {
        img.style.cursor = "zoom-in";
        img.addEventListener("click", () => {
            if (modal && modalImg) {
                modal.classList.add("is-active");
                modalImg.src = img.src;
                modalImg.alt = img.alt || "Enlarged image";
            }
        });
    });

    if (modal) {
        const closeBtn = modal.querySelector(".modal-close");
        const background = modal.querySelector(".modal-background");

        const closeModal = () => modal.classList.remove("is-active");
        if (closeBtn) closeBtn.addEventListener("click", closeModal);
        if (background) background.addEventListener("click", closeModal);

        // Optional: close on Escape key
        document.addEventListener("keydown", e => {
            if (e.key === "Escape") closeModal();
        });
    }
})

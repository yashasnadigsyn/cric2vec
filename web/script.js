document.addEventListener('DOMContentLoaded', () => {
    // Scroll Animation
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, observerOptions);

    document.querySelectorAll('.fade-in').forEach(el => observer.observe(el));

    // Stick Navbar Blur
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.style.background = 'rgba(5, 5, 16, 0.95)';
        } else {
            navbar.style.background = 'rgba(5, 5, 16, 0.8)';
        }
    });

    // Tab Switching for Inference Demo
    const tabs = document.querySelectorAll('.tab-btn');
    const contents = document.querySelectorAll('.tab-content');

    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all
            tabs.forEach(t => t.classList.remove('active'));
            contents.forEach(c => c.style.display = 'none');

            // Add active to clicked
            tab.classList.add('active');
            const targetId = tab.getAttribute('data-tab');
            document.getElementById(targetId).style.display = 'block';
        });
    });

    // Simulate Typing Effect for Demo
    const demoTexts = {
        'sim-output': `> Simulating V Kohli vs JO Holder using Crick2Vec...
        
Ball 1: 0_run (Dot Ball)
Ball 2: 1_run (Single)
Ball 3: 4_run (Boundary!)
Ball 4: 0_run (Play and miss)
Ball 5: 2_run (Double)
Ball 6: 6_run (MAXIMUM!)

Over Summary: 13 runs / 0 wickets
Predicted Outcome: High Probability of Aggression`,
        'bunny-output': `> Finding "Bunnies" for Jasprit Bumrah...

1. GJ Maxwell (Prob: 0.32)
2. AB de Villiers (Prob: 0.28)
3. KA Pollard (Prob: 0.25)
4. AD Russell (Prob: 0.24)

Analysis: Bumrah's embedding vector shows high orthogonality 
to aggressive power-hitters defined in the subspace.`
    };

    function typeWriter(elementId, text) {
        const el = document.getElementById(elementId);
        if (!el) return;
        el.textContent = '';
        let i = 0;
        function type() {
            if (i < text.length) {
                el.textContent += text.charAt(i);
                i++;
                setTimeout(type, 20); // typing speed
            }
        }
        type();
    }

    // Initialize first tab
    document.getElementById('sim-demo').style.display = 'block';
});

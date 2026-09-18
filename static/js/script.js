// SmartCart - Interactive UI Controller

document.addEventListener('DOMContentLoaded', () => {
    console.log('SmartCart Application Loaded Successfully.');
    
    // Highlight active navigation link based on window location
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-link');
    
    navLinks.forEach(link => {
        const linkPath = link.getAttribute('href');
        if (currentPath === linkPath || (currentPath === '/' && linkPath === '/')) {
            link.classList.add('active');
        } else if (currentPath !== '/' && linkPath !== '/' && currentPath.startsWith(linkPath)) {
            link.classList.add('active');
        }
    });
});

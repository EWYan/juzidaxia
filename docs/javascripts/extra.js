// 自定义JavaScript功能

document.addEventListener('DOMContentLoaded', function() {
    // 添加文章阅读时间估算
    function estimateReadingTime() {
        const article = document.querySelector('.md-content__inner');
        if (!article) return;
        
        const text = article.textContent || article.innerText;
        const wordCount = text.trim().split(/\s+/).length;
        const readingTime = Math.ceil(wordCount / 200); // 假设200字/分钟
        
        const readingTimeElement = document.createElement('div');
        readingTimeElement.className = 'reading-time';
        readingTimeElement.innerHTML = `📖 预计阅读时间: ${readingTime} 分钟`;
        readingTimeElement.style.cssText = `
            color: var(--md-default-fg-color--light);
            font-size: 0.8rem;
            margin: 1rem 0;
            padding: 0.5rem;
            border-left: 3px solid var(--md-primary-fg-color);
            background: var(--md-default-bg-color--light);
        `;
        
        const title = article.querySelector('h1');
        if (title) {
            title.parentNode.insertBefore(readingTimeElement, title.nextSibling);
        }
    }
    
    // 添加代码复制按钮增强
    function enhanceCodeBlocks() {
        document.querySelectorAll('pre > code').forEach(function(codeBlock) {
            const pre = codeBlock.parentNode;
            if (!pre.classList.contains('no-copy')) {
                const button = document.createElement('button');
                button.className = 'copy-code-button';
                button.innerHTML = '📋';
                button.title = '复制代码';
                button.style.cssText = `
                    position: absolute;
                    top: 0.5rem;
                    right: 0.5rem;
                    background: var(--md-default-bg-color);
                    border: 1px solid var(--md-default-fg-color--lightest);
                    border-radius: 0.25rem;
                    padding: 0.25rem 0.5rem;
                    cursor: pointer;
                    font-size: 0.8rem;
                    opacity: 0.7;
                    transition: opacity 0.3s ease;
                `;
                
                button.addEventListener('mouseenter', function() {
                    this.style.opacity = '1';
                });
                
                button.addEventListener('mouseleave', function() {
                    this.style.opacity = '0.7';
                });
                
                button.addEventListener('click', function() {
                    const text = codeBlock.textContent;
                    navigator.clipboard.writeText(text).then(function() {
                        const originalHTML = button.innerHTML;
                        button.innerHTML = '✅';
                        button.style.color = 'green';
                        setTimeout(function() {
                            button.innerHTML = originalHTML;
                            button.style.color = '';
                        }, 2000);
                    });
                });
                
                pre.style.position = 'relative';
                pre.appendChild(button);
            }
        });
    }
    
    // 添加平滑滚动
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            const targetId = this.getAttribute('href');
            if (targetId === '#') return;
            
            const targetElement = document.querySelector(targetId);
            if (targetElement) {
                e.preventDefault();
                targetElement.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });
    
    // 添加图片灯箱效果
    document.querySelectorAll('.md-content img').forEach(img => {
        img.style.cursor = 'zoom-in';
        img.addEventListener('click', function() {
            const overlay = document.createElement('div');
            overlay.style.cssText = `
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0,0,0,0.9);
                display: flex;
                justify-content: center;
                align-items: center;
                z-index: 9999;
                cursor: zoom-out;
            `;
            
            const enlargedImg = document.createElement('img');
            enlargedImg.src = this.src;
            enlargedImg.alt = this.alt;
            enlargedImg.style.cssText = `
                max-width: 90%;
                max-height: 90%;
                object-fit: contain;
                border-radius: 0.5rem;
            `;
            
            overlay.appendChild(enlargedImg);
            document.body.appendChild(overlay);
            
            overlay.addEventListener('click', function() {
                document.body.removeChild(overlay);
            });
        });
    });
    
    // 初始化功能
    estimateReadingTime();
    enhanceCodeBlocks();
    
    console.log('juzidaxia博客自定义脚本已加载');
});

// 添加页面加载进度条
(function() {
    const progressBar = document.createElement('div');
    progressBar.id = 'page-progress';
    progressBar.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        width: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--md-primary-fg-color), var(--md-accent-fg-color));
        z-index: 99999;
        transition: width 0.3s ease;
    `;
    document.body.appendChild(progressBar);
    
    window.addEventListener('load', function() {
        progressBar.style.width = '100%';
        setTimeout(() => {
            progressBar.style.opacity = '0';
            setTimeout(() => {
                document.body.removeChild(progressBar);
            }, 300);
        }, 300);
    });
})();
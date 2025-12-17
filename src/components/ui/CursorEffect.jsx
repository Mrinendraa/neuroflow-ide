import React, { useEffect, useState, useRef } from 'react';
import { useTheme } from '../../context/ThemeContext';
import './CursorEffect.css';

const CursorEffect = () => {
    const { theme } = useTheme();
    const cursorRef = useRef(null);
    const trailerRef = useRef(null);

    useEffect(() => {
        const cursor = cursorRef.current;
        const trailer = trailerRef.current;

        if (!cursor || !trailer) return;

        const moveCursor = (e) => {
            const { clientX, clientY } = e;

            // Move the main cursor immediately
            cursor.style.transform = `translate(${clientX}px, ${clientY}px)`;

            // Move the trailer with a slight delay/smoothing (handled by CSS transition or requestAnimationFrame)
            // For smoother performance, we use direct DOM manipulation here
            trailer.animate({
                transform: `translate(${clientX - 10}px, ${clientY - 10}px)`
            }, {
                duration: 500,
                fill: "forwards"
            });
        };

        const handleMouseDown = () => {
            cursor.classList.add('expand');
            trailer.classList.add('expand');
        };

        const handleMouseUp = () => {
            cursor.classList.remove('expand');
            trailer.classList.remove('expand');
        };

        window.addEventListener('mousemove', moveCursor);
        window.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mouseup', handleMouseUp);

        return () => {
            window.removeEventListener('mousemove', moveCursor);
            window.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mouseup', handleMouseUp);
        };
    }, []);

    return (
        <>
            <div className={`custom-cursor ${theme}`} ref={cursorRef} />
            <div className={`cursor-trailer ${theme}`} ref={trailerRef} />
        </>
    );
};

export default CursorEffect;

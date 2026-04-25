# Dashboard improvement list

## Critical bugs (must fix)
1. Map hover tooltip uses plotly_hover with scattergl which is unreliable — need custom mousemove-based point detection or switch to scatter (non-gl) for <5k points
2. 96MB garment_neighbors.json loaded eagerly — browser may hang on slow connections
3. No loading state — page shows blank/broken while 100MB+ of JSON loads
4. Outfit builder axis comparison uses hardcoded axis range (0.3) — needs to use actual data range
5. No error handling if any JSON fetch fails
6. Map container has fixed height but no responsive adaptation
7. Tab panels don't preserve scroll position
8. Selected outfit state is lost when switching tabs
9. Garment chip click area too small on mobile
10. No fallback for when images fail to load

## Visual design problems
11. Map dots are 4px — too small to see individual points, looks like noise
12. Map colors for clusters are hard to distinguish (Portland colorscale on dark bg)
13. No visual connection between map selection and other panels
14. Cards all look identical — no visual hierarchy between query and results
15. Axis strip items too small (56px) — can't see the outfits
16. Garment chips are plain boxes with no context
17. The header glow effect is barely visible and adds nothing
18. Font sizes too inconsistent — jumping between 9px-46px
19. Sidebar feels disconnected from main content
20. No visual feedback when clicking/selecting things
21. Footer is boring boilerplate
22. Stat numbers in header have no visual impact
23. The "→" arrow in outfit decomposition is ugly plain text
24. No image preloading — thumbnails pop in one by one
25. Grid layouts feel generic/template-like
26. Color palette is safe/boring — needs more character
27. No texture or depth — everything is flat boxes
28. Border radius inconsistent
29. Card hover effects are generic (translateY)
30. No transitions between content changes

## Missing features — map
31. No ability to zoom to a cluster
32. No selected point highlight on map (red star or ring)
33. No neighbor connections shown on map (lines to nearest neighbors)
34. No lasso selection to explore a region
35. No search/filter by caption text
36. Map should show cluster boundaries or density contours
37. No minimap or zoom indicator
38. Can't click-drag to explore — only plotly's built-in pan
39. No map legend for cluster colors
40. Should show cluster exemplar thumbnails on map at low zoom

## Missing features — axes
41. No axis comparison view (all 6 axes as horizontal bars for selected outfit)
42. The strip should show where the selected outfit falls
43. No ability to filter the map by axis range
44. Axis prompt text truncated on narrow screens
45. No axis correlation visualization (do axes agree or conflict?)
46. Should show distribution histogram for each axis
47. Strip labels are reversed (left should be low, right should be high)
48. No smooth scroll animation when axis changes
49. Top/bottom examples should show outfit thumbnails, not just scores
50. Should be able to sort the entire grid by an axis

## Missing features — neighbors
51. No visual distinction between query and results
52. Should show similarity score more prominently
53. Should show what's similar (shared clusters, axis values)
54. No ability to adjust k (number of neighbors)
55. Should show neighbors on the map (highlight them)
56. No "explore from here" — click a neighbor to make it the new query
57. Actually — clicking does work, but there's no visual breadcrumb trail
58. Should show garment-level overlap between query and neighbor outfits

## Missing features — garments
59. Garment crops are tiny (72px) — need larger view on click
60. No garment category filter
61. Cross-category results should show the outfit context (thumbnail of source outfit)
62. Should be able to compare garments side-by-side
63. No visual grouping by category
64. Confidence scores aren't visually meaningful — use color or opacity
65. Should highlight which garments are common/rare across the dataset
66. No garment embedding map (separate UMAP of garment crops)

## Missing features — outfit builder
67. No visual composition preview (garments arranged in outfit layout)
68. Can't add garments by browsing — only from garment detail
69. Should suggest compatible items for empty slots
70. No style coherence score for the built outfit
71. Should show the built outfit's position in embedding space
72. No undo/clear all
73. Should remember builder state across sessions (localStorage)

## Missing features — general
74. No keyboard navigation (arrow keys to browse outfits)
75. No URL hash state (can't share a link to a specific outfit)
76. No full-screen image view / lightbox
77. No data download (selected outfit, neighbors, garments)
78. No comparison mode (side-by-side two outfits)
79. No favorites / bookmark system
80. No dataset statistics visualization (category distribution, score histograms)
81. No pipeline explanation panel (how was this data made?)
82. No model comparison (CLIP vs FashionCLIP if available)
83. No search by caption
84. No "random" button to discover outfits
85. No "surprise me" / serendipity feature

## Performance
86. 96MB garment_neighbors.json — should lazy-load or paginate
87. No image lazy-loading for off-screen thumbnails
88. No debouncing on hover events
89. Should use IntersectionObserver for scroll-triggered image loading
90. Plotly bundle is 3.5MB — could use a lighter charting library
91. No service worker for offline capability
92. JSON parsing blocks the main thread — should use Web Workers

## Polish / portfolio readiness
93. No page load animation / entrance effect
94. No smooth transitions when changing tabs or selecting outfits
95. Needs a "how it works" or "about" section explaining the pipeline
96. Should have a guided tour / walkthrough for first-time visitors
97. No screenshot-ready state (everything should look good at 1440x900)
98. Needs a subtle grid or dot pattern background for depth
99. Should have a "made with" credit line that's actually interesting
100. The overall impression should be "someone spent weeks on this" not "AI generated this in 5 minutes"

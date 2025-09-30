from _spirale_src.step1_rs.random_search_sacks_points_v3 import get_points # zmieniłem nazwę na v3

img = "/home/admin2/Documents/repos/pwpw/_spirale/inputs/PXL_20250925_061456317_cut.jpg"
p = {
        'clahe_clip': 3.71,
        'clahe_grid': 12,
        'gauss_blur': 3,
        'method': 'GAUSS',
        'thresh_type': 'BIN_INV',
        'adapt_block': 20,
        'adapt_C': 10,
        'morph_open': 3,
        'morph_close': 1,
        'min_area': 25,
        'max_area': 1750,
        'min_circularity': 0.67,
        'max_elongation': 1.83,
        'dot_radius': 6,
    }
pts = get_points(img, p)  # -> List[Tuple[int,int]]

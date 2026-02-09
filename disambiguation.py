from pypinyin import lazy_pinyin
import json, re

author_data = {}
authors = set()

def latin_to_english(text):
    # 定义一个字典，将拉丁字符映射到英文字符
    translations = {
    'À':'A','Á':'A','Â':'A','Ã':'A','Ä':'A','Å':'A','Æ':'AE','Ç':'C','È':'E','É':'E',
    'Ê':'E','Ë':'E','Ì':'I','Í':'I','Î':'I','Ï':'I','Ð':'D','Ñ':'N','Ò':'O','Ó':'O',
    'Ô':'O','Õ':'O','Ö':'O','×':'x','Ø':'O','Ù':'U','Ú':'U','Û':'U','Ü':'U','Ý':'Y',
    'Þ':'TH','ß':'ss','à':'a','á':'a','â':'a','ã':'a','ä':'a','å':'a','æ':'ae','ç':'c',
    'è':'e','é':'e','ê':'e','ë':'e','ì':'i','í':'i','î':'i','ï':'i','ð':'d','ñ':'n',
    'ò':'o','ó':'o','ô':'o','õ':'o','ö':'o','ø':'o','ù':'u','ú':'u','û':'u','ü':'u',
    'ý':'y','þ':'th','ÿ':'y',

    # 2) 拉丁扩展-A (U+0100–U+017F)
    'Ā':'A','ā':'a','Ă':'A','ă':'a','Ą':'A','ą':'a','Ć':'C','ć':'c','Ĉ':'C','ĉ':'c',
    'Ċ':'C','ċ':'c','Č':'C','č':'c','Ď':'D','ď':'d','Đ':'D','đ':'d','Ē':'E','ē':'e',
    'Ĕ':'E','ĕ':'e','Ė':'E','ė':'e','Ę':'E','ę':'e','Ě':'E','ě':'e','Ĝ':'G','ĝ':'g',
    'Ğ':'G','ğ':'g','Ġ':'G','ġ':'g','Ģ':'G','ģ':'g','Ĥ':'H','ĥ':'h','Ħ':'H','ħ':'h',
    'Ĩ':'I','ĩ':'i','Ī':'I','ī':'i','Ĭ':'I','ĭ':'i','Į':'I','į':'i','İ':'I','ı':'i',
    'Ĳ':'IJ','ĳ':'ij','Ĵ':'J','ĵ':'j','Ķ':'K','ķ':'k','ĸ':'k','Ĺ':'L','ĺ':'l','Ļ':'L',
    'ļ':'l','Ľ':'L','ľ':'l','Ŀ':'L','ŀ':'l','Ł':'L','ł':'l','Ń':'N','ń':'n','Ņ':'N',
    'ņ':'n','Ň':'N','ň':'n','ŉ':'n','Ŋ':'NG','ŋ':'ng','Ō':'O','ō':'o','Ŏ':'O','ŏ':'o',
    'Ő':'O','ő':'o','Œ':'OE','œ':'oe','Ŕ':'R','ŕ':'r','Ŗ':'R','ŗ':'r','Ř':'R','ř':'r',
    'Ś':'S','ś':'s','Ŝ':'S','ŝ':'s','Ş':'S','ş':'s','Š':'S','š':'s','Ţ':'T','ţ':'t',
    'Ť':'T','ť':'t','Ŧ':'T','ŧ':'t','Ũ':'U','ũ':'u','Ū':'U','ū':'u','Ŭ':'U','ŭ':'u',
    'Ů':'U','ů':'u','Ű':'U','ű':'u','Ų':'U','ų':'u','Ŵ':'W','ŵ':'w','Ŷ':'Y','ŷ':'y',
    'Ÿ':'Y','Ź':'Z','ź':'z','Ż':'Z','ż':'z','Ž':'Z','ž':'z',

    # 3) 拉丁扩展-B 常用字符 (U+0180–U+024F)
    'Ə':'E','ə':'e','Ɛ':'E','ɛ':'e','Ƒ':'F','ƒ':'f','Ɠ':'G','Ɣ':'G','ƕ':'hv',
    'Ɩ':'I','ɪ':'i','Ɨ':'I','ɨ':'i','Ƙ':'K','ƙ':'k','ƚ':'l','Ɯ':'M','ɯ':'m',
    'Ɲ':'N','ɲ':'n','Ɵ':'O','ɵ':'o','Ƥ':'P','ƥ':'p','Ʀ':'R','ʀ':'r','Ʃ':'S',
    'ʃ':'s','Ƭ':'T','ƭ':'t','Ʈ':'T','ʈ':'t','Ư':'U','ư':'u','Ʊ':'U','ʊ':'u',
    'Ʋ':'V','ʋ':'v','Ƴ':'Y','ƴ':'y','Ƶ':'Z','ƶ':'z',

    # 4) 拉丁扩展-C/D/E (U+2C60–U+A7FF) 常用连字/划线
    'Ⱡ':'L','ⱡ':'l','Ɫ':'L','Ᵽ':'P','Ɽ':'R','ⱥ':'a','ⱦ':'t','Ⱨ':'H',
    'ⱨ':'h','Ⱪ':'K','ⱪ':'k','Ⱬ':'Z','ⱬ':'z','Ɑ':'A','ɑ':'a','Ɱ':'M','Ɐ':'A',
    'Ɒ':'O','ⱱ':'v','Ⱳ':'W','ⱳ':'w','ⱴ':'v','Ⱶ':'H','ⱶ':'h','ⱷ':'o','ⱸ':'e',
    'ⱹ':'r','ⱺ':'o','ⱻ':'e','ⱼ':'j','ⱽ':'v','Ȿ':'s','Ɀ':'z','ꜰ':'f','ꜱ':'s',

    # 5) 额外连字/斜线/钩子
    'Ǳ':'DZ','ǲ':'Dz','ǳ':'dz','Ǆ':'DŽ','ǅ':'Dž','ǆ':'dž','Ǉ':'LJ','ǈ':'Lj',
    'ǉ':'lj','Ǌ':'NJ','ǋ':'Nj','ǌ':'nj','Ǎ':'A','ǎ':'a','Ǐ':'I','ǐ':'i','Ǒ':'O',
    'ǒ':'o','Ǔ':'U','ǔ':'u','Ǖ':'U','ǖ':'u','Ǘ':'U','ǘ':'u','Ǚ':'U','ǚ':'u',
    'Ǜ':'U','ǜ':'u','ǝ':'e','Ǟ':'A','ǟ':'a','Ǡ':'A','ǡ':'a','Ǣ':'AE','ǣ':'ae',
    'Ǥ':'G','ǥ':'g','Ǧ':'G','ǧ':'g','Ǩ':'K','ǩ':'k','Ǫ':'O','ǫ':'o','Ǭ':'O',
    'ǭ':'o','Ǯ':'Z','ǯ':'z','ǰ':'j','Ǳ':'DZ','ǲ':'Dz','ǳ':'dz','Ǵ':'G','ǵ':'g',
    'Ƕ':'HV','Ƿ':'W','Ǹ':'N','ǹ':'n','Ǻ':'A','ǻ':'a','Ǽ':'AE','ǽ':'ae','Ǿ':'O',
    'ǿ':'o','Ȁ':'A','ȁ':'a','Ȃ':'A','ȃ':'a','Ȅ':'E','ȅ':'e','Ȇ':'E','ȇ':'e',
    'Ȉ':'I','ȉ':'i','Ȋ':'I','ȋ':'i','Ȍ':'O','ȍ':'o','Ȏ':'O','ȏ':'o','Ȑ':'R',
    'ȑ':'r','Ȓ':'R','ȓ':'r','Ȕ':'U','ȕ':'u','Ȗ':'U','ȗ':'u','Ș':'S','ș':'s',
    'Ț':'T','ț':'t','Ȝ':'Y','ȝ':'y','Ȟ':'H','ȟ':'h','Ƞ':'N','ȡ':'d','Ȣ':'OU',
    'ȣ':'ou','Ȥ':'Z','ȥ':'z','Ȧ':'A','ȧ':'a','Ȩ':'E','ȩ':'e','Ȫ':'O','ȫ':'o',
    'Ȭ':'O','ȭ':'o','Ȯ':'O','ȯ':'o','Ȱ':'O','ȱ':'o','Ȳ':'Y','ȳ':'y'
}
    # 使用字典进行字符替换
    for latin_char, english_char in translations.items():
        text = text.replace(latin_char, english_char)
    return text

def judge_possible_sol(possible_solutions, target_name):
    """
    Judge if the target name is in the possible solutions.
    """
    def preprocess_name(name:str):
        name = latin_to_english(name.strip())
        name = re.sub(r'ç', 'c', name)
        name = re.sub(r'ş', 's', name)
        name = re.sub(r'ı', 'i', name)
        name = re.sub(r'ü', 'u', name)
        name = re.sub(r'ö', 'o', name)
        name = re.sub(r'[.,;:!?()"\'-]', '', name)
        if any('\u4e00' <= char <= '\u9fff' for char in name):
            if not ' ' in name:
                name_parts = lazy_pinyin(name)
                name = ''.join(name_parts[1:]) + ' ' + name_parts[0]
            else:
                name = ' '.join(lazy_pinyin(name))
        return name.lower()
    target_name = preprocess_name(target_name)
    solutions_board = []
    for sol in possible_solutions:
        sol_parts = preprocess_name(sol).split()
        target_parts = target_name.split()
        score = 0
        for part in sol_parts:
            if part in target_parts:
                score += 1
        solutions_board.append((score, sol))
    solutions_board.sort(reverse=True, key=lambda x: x[0])
    highest_score = solutions_board[0][0]
    if highest_score == 0:
        print(f"No Matched Results: {possible_solutions}. Target name is '{target_name}'")
        return False, None
    best_solutions = [sol for score, sol in solutions_board if score == highest_score]
    if len(best_solutions) == 1:
        return True, best_solutions[0]
    elif len(best_solutions) > 1: # 如果有多个最优解，进行模糊匹配
        possible_sols = []
        target_parts = target_name.split()
        for sol in best_solutions:
            sol_parts = preprocess_name(sol).split()
            flag = True
            if len(sol_parts) != len(target_parts):
                continue
            for p1, p2 in zip(sol_parts, target_parts):
                if p1[0] != p2[0] or len(p1) != len(p2):
                    flag = False
                    break
            if flag:
                possible_sols.append(sol)
        if len(possible_sols) == 1:
            return True, possible_sols[0]
        elif len(possible_sols) == 0:
            # 尝试中文名调转姓名位置
            if len(target_parts) == 2:
                reversed_target_parts = target_parts[::-1]
                for sol in best_solutions:
                    sol_parts = preprocess_name(sol).split()
                    flag = True
                    for p1, p2 in zip(sol_parts, reversed_target_parts):
                        if p1[0] != p2[0] or len(p1) != len(p2):
                            flag = False
                            break
                    if flag:
                       possible_sols.append(sol)
                if len(possible_sols) == 1:
                    return True, possible_sols[0]
                elif len(possible_sols) == 0:
                    print(f"No Matched Results: {best_solutions}. Target name is '{target_name}'")
                    return False, None
            def judge_multiple_results(possible_sols, thereshold=3):
                """
                Judge if there are multiple results with low trust.
                """
                new_possible_sols = [' '.join(sorted(preprocess_name(possible_sol).split())) for possible_sol in possible_sols]
                multi_board = {}
                for sol in new_possible_sols:
                    if sol not in multi_board:
                        multi_board[sol] = 0
                    multi_board[sol] += 1
                highest_score = max(multi_board.values())
                if highest_score >= thereshold and list(multi_board.values()).count(highest_score) == 1:
                    final_solution = [sol for sol, score in multi_board.items() if score == highest_score][0]
                    return True, [possible_sol for possible_sol in possible_sols if ' '.join(sorted(preprocess_name(possible_sol).split())) == final_solution][0]
                else:
                    return False, None
            print(best_solutions)
            flag, best_solutions = judge_multiple_results(best_solutions)
            if flag:
                return True, best_solutions
            # 尝试中文名翻转和多作者取出现大于3次的都没找到
            return False, None
        else: # 如果有多个可能的结果, 可能是真的一模一样
            print(f"Multiple matching names found: {possible_sols} and cannot be disambiguated. Target name is '{target_name}'")
            return True, possible_sols
    else: # 没有找到匹配的结果
        print(f"No Matched Best Results: {best_solutions}. Target name is '{target_name}'")
        return False, None
    
def judge_same(title1:str, title2:str):
    """
    判断两个标题是否相同。
    :param title1: 第一个标题
    :param title2: 第二个标题
    :return: 如果相同，返回 True；否则返回 False
    """
    # 去除所有标点符号和多余的空格
    import re
    title1 = re.sub(r'[^\w\s]', '', title1).strip()
    title2 = re.sub(r'[^\w\s]', '', title2).strip()
    title1 = title1.split()
    title2 = title2.split()
    for t1, t2 in zip(title1, title2):
        if t1.lower() != t2.lower():
            return False
    return True

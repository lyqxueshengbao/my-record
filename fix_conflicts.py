import os
import re


def fix_conflict_in_file(filepath):
    """清理单个文件中的冲突标记，保留 main 分支（theirs）的版本"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # 检查是否有冲突标记
        if '<<<<<<< HEAD' not in content:
            return False

        # 正则模式：匹配整个冲突块，保留 ======= 和 >>>>>>> 之间的内容
        # <<<<<<< HEAD
        # ... (删除这部分)
        # =======
        # ... (保留这部分)
        # >>>>>>> xxx

        pattern = r'<<<<<<< HEAD\n(.*?)\n=======\n(.*?)\n>>>>>>> [^\n]+\n?'

        # 替换为保留的内容（第二组）
        fixed_content = re.sub(pattern, r'\2\n', content, flags=re.DOTALL)

        # 写回文件
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(fixed_content)

        print(f"✓ 已修复: {filepath}")
        return True

    except Exception as e:
        print(f"✗ 错误 {filepath}: {e}")
        return False


def main():
    """扫描并修复所有有冲突的文件"""
    fixed_count = 0

    # 这些是你的冲突文件
    conflict_files = [
        'configs/carrada/config_mvrecordoi_carrada.yaml',
        'configs/carrada/config_record_ra_carrada.yaml',
        'configs/carrada/config_record_rd_carrada.yaml',
        'configs/carrada/config_recordoi_ra_carrada.yaml',
        'configs/carrada/config_recordoi_rd_carrada.yaml',
        'configs/cruw/config_record_cruw_nolstm_multi.yaml',
        'configs/cruw/config_record_cruw_nolstm_single.yaml',
        'configs/cruw/config_recordoi_cruw.yaml',
        'main_carrada.py'
    ]

    for filepath in conflict_files:
        if os.path.exists(filepath):
            if fix_conflict_in_file(filepath):
                fixed_count += 1
        else:
            print(f"⚠ 文件不存在: {filepath}")

    print(f"\n总共修复了 {fixed_count} 个文件")


if __name__ == '__main__':
    main()

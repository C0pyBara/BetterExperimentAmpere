import os
import json


def json_to_html_table(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    table = data[0]["cells"]

    # Размер таблицы
    max_row = max(max(cell["row_nums"]) for cell in table)
    max_col = max(max(cell["column_nums"]) for cell in table)

    n_rows = max_row + 1
    n_cols = max_col + 1

    # Grid
    grid = [[None for _ in range(n_cols)] for _ in range(n_rows)]

    # Заполнение
    for cell in table:
        rows = cell["row_nums"]
        cols = cell["column_nums"]

        r0, c0 = rows[0], cols[0]

        rowspan = len(rows)
        colspan = len(cols)

        grid[r0][c0] = {
            "text": cell["xml_text_content"],
            "rowspan": rowspan,
            "colspan": colspan,
            "is_header": cell["is_column_header"]
        }

        # помечаем span
        for r in rows:
            for c in cols:
                if (r, c) != (r0, c0):
                    grid[r][c] = "SPAN"

    # Генерация HTML
    html = "<table border='1' cellspacing='0' cellpadding='5'>\n"

    for r in range(n_rows):
        html += "  <tr>\n"
        c = 0
        while c < n_cols:
            cell = grid[r][c]

            # 🔥 ВАЖНО: если пусто → вставляем пустую ячейку
            if cell is None:
                html += "    <td></td>\n"
                c += 1
                continue

            # пропускаем span-зону
            if cell == "SPAN":
                c += 1
                continue

            tag = "th" if cell["is_header"] else "td"

            attrs = ""
            if cell["rowspan"] > 1:
                attrs += f" rowspan='{cell['rowspan']}'"
            if cell["colspan"] > 1:
                attrs += f" colspan='{cell['colspan']}'"

            html += f"    <{tag}{attrs}>{cell['text']}</{tag}>\n"

            # 🔥 пропускаем colspan
            c += cell["colspan"]

        html += "  </tr>\n"

    html += "</table>"

    return html


def convert_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for file_name in os.listdir(input_folder):
        if not file_name.endswith(".json"):
            continue

        input_path = os.path.join(input_folder, file_name)

        try:
            html = json_to_html_table(input_path)

            output_file = file_name.replace(".json", ".html")
            output_path = os.path.join(output_folder, output_file)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            print(f"✔ {file_name}")

        except Exception as e:
            print(f"✖ {file_name}: {e}")


# === запуск ===
convert_folder(r"C:\Users\Юзя\Desktop\BetterExperimentAmpere\Get_500_Tables_from_PubTables\JSON_Complex_TOP500_normalized", r"C:\Users\Юзя\Desktop\BetterExperimentAmpere\Get_500_Tables_from_PubTables\JSON_Complex_TOP500_normalized_html")
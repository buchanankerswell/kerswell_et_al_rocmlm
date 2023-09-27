from draft import parse_arguments, check_arguments, write_markdown_tables, replace_placeholders

# Parse arguments and check
args = parse_arguments()
valid_args = check_arguments(args, "place-tables.py")

# Load valid arguments
locals().update(valid_args)

# Write markdown tables
write_markdown_tables()

# Place markdown tables in manuscript
replace_placeholders(f"{ms}.md", "tmp.md")

print("place-tables.py done!")
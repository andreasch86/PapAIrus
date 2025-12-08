from pathlib import Path

class ContextGatherer:
    def __init__(self, setting):
        self.setting = setting

    def gather(self):
        project_root = self.setting.project.target_repo

        context = {
            "project_name": project_root.name,
            "entry_point_summary": "Entry point analysis pending.",
            "usage_context_from_tests": "Integration tests analysis pending.",
            "tests_map": {}
        }

        # 1. README
        readme_path = project_root / "README.md"
        if readme_path.exists():
            context["readme"] = readme_path.read_text(errors="ignore")

        # 2. Entry point
        for name in ["main.py", "app.py", "manage.py", "cli.py"]:
            p = project_root / name
            if p.exists():
                context["entry_point_summary"] = f"Entry point ({name}):\n" + p.read_text(errors="ignore")[:2000]
                break

        # 3. Tests
        tests_path = project_root / "tests"
        if tests_path.exists():
            integration_tests = list(tests_path.rglob("*integration*/*.py"))
            unit_tests = list(tests_path.rglob("test_*.py"))

            # Usage context from integration tests
            usage_context = []
            for t in integration_tests[:3]:
                 usage_context.append(f"--- {t.name} ---\n{t.read_text(errors='ignore')[:2000]}")

            if usage_context:
                context["usage_context_from_tests"] = "\n".join(usage_context)
            else:
                 # Fallback to some unit tests if no integration tests
                 for t in unit_tests[:3]:
                     usage_context.append(f"--- {t.name} ---\n{t.read_text(errors='ignore')[:2000]}")
                 context["usage_context_from_tests"] = "\n".join(usage_context)

            # Map for specific file tests
            for t in integration_tests + unit_tests:
                context["tests_map"][t.name] = t.read_text(errors="ignore")

        # 4. Existing Docs Analysis (Create if missing)
        docs_path = project_root / self.setting.project.markdown_docs_name
        if not docs_path.exists():
            docs_path.mkdir(parents=True, exist_ok=True)
        else:
            md_files = sorted(list(docs_path.rglob("*.md")))
            if md_files:
                docs_sample = []
                for md in md_files[:2]:
                    docs_sample.append(f"--- {md.name} ---\n{md.read_text(errors='ignore')[:1000]}")
                if docs_sample:
                    context["existing_docs_sample"] = "\n".join(docs_sample)

        return context

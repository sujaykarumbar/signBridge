import app.config.firebase_admin_init  # noqa: F401
from firebase_admin import db
from datetime import datetime
import uuid

_in_memory_history = {}


def _history_ref(user_id):
    try:
        return db.reference(f"user/{user_id}/{user_id}/history")
    except Exception:
        return None


def retrieve_history(user_id):
    ref = _history_ref(user_id)
    if ref is None:
        user_list = _in_memory_history.get(user_id, [])
        return sorted(user_list, key=lambda x: x["timestamp"], reverse=True)

    data = ref.get()

    if data is None:
        return []

    # New format: dictionary of Firebase push IDs
    if isinstance(data, dict):
        history = []

        for item_id, item in data.items():
            history.append({
                "id": item_id,
                "translation": item.get("translation", ""),
                "timestamp": item.get("timestamp", "")
            })

        # Newest first
        history.sort(
            key=lambda x: x["timestamp"],
            reverse=True
        )

        return history

    return []


def store_translation(user_id, translation):
    ref = _history_ref(user_id)
    if ref is None:
        item = {
            "id": str(uuid.uuid4())[:8],
            "translation": translation,
            "timestamp": datetime.now().isoformat(),
        }
        _in_memory_history.setdefault(user_id, []).append(item)
        return item

    push_ref = ref.push()

    item = {
        "translation": translation,
        "timestamp": datetime.now().isoformat()
    }

    push_ref.set(item)

    return {
        "id": push_ref.key,
        **item
    }


def delete_translation(user_id, translation_id):
    ref = _history_ref(user_id)
    if ref is None:
        user_list = _in_memory_history.get(user_id, [])
        for i, item in enumerate(user_list):
            if item["id"] == translation_id:
                user_list.pop(i)
                return True
        return False

    child_ref = ref.child(translation_id)

    if child_ref.get() is None:
        return False

    child_ref.delete()
    return True

def delete_all_translations(user_id):
    ref = _history_ref(user_id)
    if ref is None:
        _in_memory_history[user_id] = []
        return True
    if ref.get() is None:
        return False
    ref.delete()
    return True
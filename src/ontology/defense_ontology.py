NODE_TYPES = {
    "Actor": ["StateActor", "NonStateActor", "FinancialInstitution", "Individual"],
    "Location": ["MilitaryBase", "UrbanArea", "CriticalInfrastructure", "BorderZone"],
    "Event": ["ConflictEvent", "EconomicEvent", "OSINTSignal"],
    "Equipment": ["Vehicle", "Weapon", "Infrastructure"],
    "ObservationSource": ["Satellite", "SocialMedia", "DeliveryService"]
}

RELATION_SCHEMA = {
    ("FinancialInstitution", "Funds", "NonStateActor"),
    ("Actor", "Initiates", "ConflictEvent"),
    ("ConflictEvent", "LocatedAt", "Location"),
    ("ConflictEvent", "Uses", "Equipment"),
    ("ObservationSource", "Reports", "Event"),
    ("Actor", "Controls", "CriticalInfrastructure"),
    ("EconomicEvent", "Affects", "Actor")
}

def is_valid_relation(source_type, rel_type, target_type):
    """Vérifie si une relation est autorisée par l'ontologie."""
    return (source_type, rel_type, target_type) in RELATION_SCHEMA

def get_node_supertype(node_type):
    """Retourne la classe parente (ex: 'FinancialInstitution' → 'Actor')."""
    for supertype, subtypes in NODE_TYPES.items():
        if node_type in subtypes:
            return supertype
    return None

from flask import Flask, request, jsonify

app = Flask(__name__)

# Simulated database
DOCTORS = {
    "cardiology": [
        {"id": 1, "name": "Dr. Heart", "available_slots": ["10:00 AM", "2:00 PM"]},
        {"id": 2, "name": "Dr. Pulse", "available_slots": ["11:00 AM", "4:00 PM"]}
    ],
    "dermatology": [
        {"id": 3, "name": "Dr. Skin", "available_slots": ["9:00 AM", "1:00 PM"]}
    ]
}

RESERVATIONS = []  # Store reservations here

@app.route('/get_doctors', methods=['GET'])
def get_doctors():
    illness = request.args.get('illness')
    doctors = DOCTORS.get(illness.lower(), [])
    if not doctors:
        return jsonify({"message": "No doctors found for the specified illness."}), 404
    return jsonify(doctors)

@app.route('/get_availability', methods=['GET'])
def get_availability():
    doctor_id = int(request.args.get('doctor_id'))
    for specialty, doctors in DOCTORS.items():
        for doctor in doctors:
            if doctor['id'] == doctor_id:
                return jsonify(doctor['available_slots'])
    return jsonify({"message": "Doctor not found."}), 404

@app.route('/make_reservation', methods=['POST'])
def make_reservation():
    data = request.json
    doctor_id = data['doctor_id']
    patient_name = data['patient_name']
    slot = data['slot']

    for specialty, doctors in DOCTORS.items():
        for doctor in doctors:
            if doctor['id'] == doctor_id:
                if slot in doctor['available_slots']:
                    doctor['available_slots'].remove(slot)
                    reservation = {
                        "doctor_id": doctor_id,
                        "patient_name": patient_name,
                        "slot": slot
                    }
                    RESERVATIONS.append(reservation)
                    return jsonify({"message": "Reservation successful", "reservation": reservation})
                else:
                    return jsonify({"message": "Slot not available."}), 400

    return jsonify({"message": "Doctor not found."}), 404

@app.route('/get_reservations', methods=['GET'])
def get_reservations():
    return jsonify(RESERVATIONS)

if __name__ == '__main__':
    app.run(debug=True)

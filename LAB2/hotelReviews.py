import skfuzzy as fuzz
from skfuzzy import control as ctrl


def create_fuzzy_system():
    # Define input and output variables
    restaurant_rating = ctrl.Antecedent(
        universe=(0, 10), label='Restaurant Rating')
    room_rating = ctrl.Antecedent(universe=(0, 10), label='Room Rating')
    service_rating = ctrl.Antecedent(universe=(0, 10), label='Service Rating')
    hotel_quality = ctrl.Consequent(universe=(0, 10), label='Hotel Quality')

    # Automatically generate membership functions for input variables
    for variable in [restaurant_rating, room_rating, service_rating]:
        variable.automf(3)

    # Define membership functions for the output variable
    hotel_quality['poor'] = fuzz.trimf(hotel_quality.universe, [0, 0, 5])
    hotel_quality['average'] = fuzz.trimf(hotel_quality.universe, [0, 5, 10])
    hotel_quality['good'] = fuzz.trimf(hotel_quality.universe, [5, 10, 10])

    return restaurant_rating, room_rating, service_rating, hotel_quality


def create_rules(restaurant_rating, room_rating, service_rating, hotel_quality):
    # Define rules without weights
    rules = [
        ctrl.Rule(restaurant_rating['poor'] | room_rating['poor']
                  | service_rating['poor'], hotel_quality['poor']),
        ctrl.Rule(service_rating['average'], hotel_quality['average']),
        ctrl.Rule(restaurant_rating['good'] | room_rating['good']
                  | service_rating['good'], hotel_quality['good'])
    ]

    return rules


def main():
    # Create the fuzzy logic system
    restaurant_rating, room_rating, service_rating, hotel_quality = create_fuzzy_system()
    rules = create_rules(restaurant_rating, room_rating,
                         service_rating, hotel_quality)

    # Create the control system
    hotel_rating_ctrl = ctrl.ControlSystem(rules)

    # Modify input values to simulate rule weights
    restaurant_rating_value = input('Enter Restaurant Rating (0-10): ')
    room_rating_value = input('Enter Room Rating (0-10): ')
    service_rating_value = input('Enter Service Rating (0-10): ')

    restaurant_rating_value, room_rating_value, service_rating_value = float(
        restaurant_rating_value), float(room_rating_value), float(service_rating_value)

    # Apply scaling factors to simulate rule weights
    restaurant_rating_value *= 0.5  # Adjust the scaling factor as needed
    room_rating_value *= 1.0  # Adjust the scaling factor as needed
    service_rating_value *= 0.7  # Adjust the scaling factor as needed

    # Create the control system simulation with modified input values
    hotel_rating = ctrl.ControlSystemSimulation(hotel_rating_ctrl)
    hotel_rating.input['Restaurant Rating'] = restaurant_rating_value
    hotel_rating.input['Room Rating'] = room_rating_value
    hotel_rating.input['Service Rating'] = service_rating_value

    # Compute the result
    hotel_rating.compute()

    # Get the output (hotel quality)
    quality = hotel_rating.output['Hotel Quality']

    # Determine the review based on the output
    if 0 <= quality <= 3:
        review = "Poor hotel quality."
    elif 3 < quality <= 7:
        review = "Average hotel quality."
    else:
        review = "Good hotel quality."

    # Print the review
    print(f"Your hotel quality rating: {quality}")
    print(f"Summary: {review}")


if __name__ == "__main__":
    main()

import pytest
from fire_behavior import fire_simulation as fs

def test_input_right():
    # test if the wind direction is in : 'N', 'S', 'E', 'W', 'NW', 'NE', 'SE', 'SW'
    a = fs.wind_fire_simulation('SE')
    assert a == "Simulation succeed!!"

def test_input_wrong():
    # test if the wind direction is not in : 'N', 'S', 'E', 'W', 'NW', 'NE', 'SE', 'SW'
    with pytest.raises(Exception):
    	a = fs.wind_fire_simulation('Dong')

import 'package:shared_preferences/shared_preferences.dart';

class SharedPreference {
  sharedPrefInit() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
  }

  Future<bool> checkValuePresent(key) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    bool checkValuePresent = prefs.containsKey('$key');
    return checkValuePresent;
  }

  saveCred({
    required String email,
    required String password,
    required String name,
    required String profilePicture,
    required String userID,
  }) async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.setString('email', email);
    prefs.setString('password', password);
    prefs.setString('name', name);
    prefs.setString('profilePicture', profilePicture);
    prefs.setString('userID', userID);
  }

  Future<void> updateCred(String key, String value) async {
    final SharedPreferences prefs = await SharedPreferences.getInstance();
    final String currentValue = prefs.getString(key) ?? '';
    if (currentValue.isNotEmpty) {
      prefs.remove(key); // remove the current value
      await prefs.setString(key, value); // save the new value with the same key
    }
  }

  Future<String?> getCred(String key) async {
    try {
      SharedPreferences prefs = await SharedPreferences.getInstance();
      final result = prefs.getString('$key');
      return result;
    } catch (e) {
      return 'Error Fetching Data';
    }
  }

  reset() async {
    SharedPreferences prefs = await SharedPreferences.getInstance();
    prefs.remove('email');
    prefs.remove('password');
  }
}

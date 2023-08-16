import 'package:flutter/material.dart';
import 'package:recipedia/Admin/recipes/update_or_delete_recipe.dart';
import 'package:recipedia/Admin/recipes/view_recipe_details.dart';

class RecipeCard extends StatelessWidget {
  final String id;
  final String title;
  final String description;
  final int rating;
  final String cookTime;
  final String thumbnailUrl;
  final String category;
  final String diet;
  final String course;
  final String ingredients;
  final bool isAdmin;

  const RecipeCard({
    super.key,
    required this.isAdmin,
    required this.id,
    required this.title,
    required this.description,
    required this.cookTime,
    required this.rating,
    required this.thumbnailUrl,
    required this.diet,
    required this.course,
    required this.ingredients,
    required this.category,
  });

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return InkWell(
      onTap: () {
        Navigator.push(
          context,
          PageRouteBuilder(
            transitionDuration: const Duration(seconds: 1),
            transitionsBuilder: (context, animation, animationTime, child) {
              animation = CurvedAnimation(
                  parent: animation, curve: Curves.fastLinearToSlowEaseIn);
              return ScaleTransition(
                scale: animation,
                alignment: Alignment.center,
                child: child,
              );
            },
            pageBuilder: (context, animation, animationTime) {
              return isAdmin
                  ? UpdateOrDeleteRecipe(
                      id: id,
                      title: title,
                      description: description,
                      cookTime: cookTime,
                      rating: rating,
                      diet: diet,
                      course: course,
                      thumbnailUrl: thumbnailUrl,
                      ingredients: ingredients,
                      category: category,
                    )
                  : ViewRecipeDetails(
                      ID: id,
                      title: title,
                      description: description,
                      cookTime: cookTime,
                      diet: diet,
                      course: course,
                      rating: rating,
                      thumbnailUrl: thumbnailUrl,
                      ingredients: ingredients,
                    );
            },
          ),
        );
      },
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 10),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ClipRRect(
              borderRadius: BorderRadius.circular(30.0),
              child: Image.network(
                thumbnailUrl,
                width: size.width * 1,
                height: 200,
                fit: BoxFit.cover,
              ),
            ),
            const SizedBox(
              height: 12.5,
            ),
            Text(
              title,
              style: const TextStyle(
                fontSize: 16,
                color: Colors.black,
              ),
              maxLines: 1,
              textAlign: TextAlign.left,
            ),
            const SizedBox(
              height: 05,
            ),
            Text(
              diet,
              style: const TextStyle(
                fontSize: 15,
                color: Colors.black54,
              ),
              maxLines: 1,
              textAlign: TextAlign.left,
            ),
          ],
        ),
      ),
    );
  }
}

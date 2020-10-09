package miniJava.interpreter.miniJava;

import org.eclipse.emf.ecore.EObject;

public interface WriteImage extends EObject, Statement {
	Expression getPath();

	void setPath(Expression value);

	Expression getImage();

	void setImage(Expression value);

	void evaluateStatement(State state);
}

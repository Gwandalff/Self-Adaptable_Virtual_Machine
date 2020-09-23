package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface WriteImage extends EObject, AcceptInterface, Statement {
	Expression getPath();

	void setPath(Expression value);

	Expression getImage();

	void setImage(Expression value);
}

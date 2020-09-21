package miniJava.visitor.miniJava;

import org.eclipse.emf.ecore.EObject;
import visitor.AcceptInterface;

public interface Frame extends EObject, AcceptInterface {
	Call getCall();

	void setCall(Call value);

	ObjectInstance getInstance();

	void setInstance(ObjectInstance value);

	Frame getChildFrame();

	void setChildFrame(Frame value);

	Frame getParentFrame();

	void setParentFrame(Frame value);

	Context getRootContext();

	void setRootContext(Context value);

	Value getReturnValue();

	void setReturnValue(Value value);
}
